//===-- iota.cpp - LLVM to Common Lisp transpiler -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Iota generates Common Lisp code compatible with the Iota runtime
// from LLVM bitcode.
//
//===----------------------------------------------------------------------===//


#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/NaCl.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/CommandLine.h"
#include <unordered_map>

using namespace llvm;

extern void lowerBitcastTrunc(Module &M);
extern void lowerVectorBswap(Module &M);
extern void eliminateLifetimeIntrinsics(Module &M);

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input file>"), cl::Required);

static cl::opt<bool>
EmitTypeDeclarartions("emit-type-declarations", cl::desc("Emit type declarations for variables"), cl::init(true));
static cl::opt<bool>
PrintLLVMInstructions("print-llvm-instructions", cl::desc("Print original LLVM instructions in comments interleaved with translated code"));

static cl::opt<std::string>
PackageName("package", cl::desc("Package name"));

static cl::opt<std::string>
OptimizeQualities("optimize", cl::desc("Optimize qualities"));

static cl::opt<std::string>
ContextPersonality("personality", cl::desc("Context personality"));

static cl::opt<std::string>
EntryPoint("entry-point", cl::desc("Symbol to use as the entry point"), cl::init("_start"));

static cl::opt<bool>
TranslateOnly("translate-only", cl::desc("Only translate input, do not run additional LLVM passes"));

static std::string escapeSymbol(StringRef name) {
    std::string result = "";
    result += "|";
    for(auto &ch: name) {
        if(ch >= 0x7F) {
            report_fatal_error("Invalid character in name " + name);
        }
        if(ch == '\\' || ch == '|') {
            result += "\\";
        }
        result += ch;
    }
    result += "|";
    return result;
}

static uint64_t resolveConstantInteger(std::unique_ptr<Module> &module, std::unordered_map<GlobalObject *, unsigned long long> &global_table, Constant *value);

static uint64_t reallyResolveConstantInteger(std::unique_ptr<Module> &module, std::unordered_map<GlobalObject *, unsigned long long> &global_table, Constant *value) {
    if(auto CE = dyn_cast<ConstantExpr>(value)) {
        if(CE->getOpcode() == Instruction::BitCast || CE->getOpcode() == Instruction::IntToPtr || CE->getOpcode() == Instruction::PtrToInt) {
            return resolveConstantInteger(module, global_table, cast<Constant>(CE->getOperand(0)));
        } else if(CE->getOpcode() == Instruction::ICmp) {
            auto lhs = resolveConstantInteger(module, global_table, cast<Constant>(CE->getOperand(0)));
            auto rhs = resolveConstantInteger(module, global_table, cast<Constant>(CE->getOperand(1)));
            if(CE->getPredicate() == ICmpInst::ICMP_EQ) {
                return lhs == rhs ? 1 : 0;
            } else if(CE->getPredicate() == ICmpInst::ICMP_NE) {
                return lhs != rhs ? 1 : 0;
            }
        } else if(CE->getOpcode() == Instruction::Add) {
            auto lhs = resolveConstantInteger(module, global_table, cast<Constant>(CE->getOperand(0)));
            auto rhs = resolveConstantInteger(module, global_table, cast<Constant>(CE->getOperand(1)));
            return lhs + rhs;
        } else if(CE->getOpcode() == Instruction::Sub) {
            auto lhs = resolveConstantInteger(module, global_table, cast<Constant>(CE->getOperand(0)));
            auto rhs = resolveConstantInteger(module, global_table, cast<Constant>(CE->getOperand(1)));
            return lhs - rhs;
        } else if(CE->getOpcode() == Instruction::Or) {
            auto lhs = resolveConstantInteger(module, global_table, cast<Constant>(CE->getOperand(0)));
            auto rhs = resolveConstantInteger(module, global_table, cast<Constant>(CE->getOperand(1)));
            return lhs | rhs;
        } else if(CE->getOpcode() == Instruction::And) {
            auto lhs = resolveConstantInteger(module, global_table, cast<Constant>(CE->getOperand(0)));
            auto rhs = resolveConstantInteger(module, global_table, cast<Constant>(CE->getOperand(1)));
            return lhs | rhs;
        } else if(CE->getOpcode() == Instruction::Xor) {
            auto lhs = resolveConstantInteger(module, global_table, cast<Constant>(CE->getOperand(0)));
            auto rhs = resolveConstantInteger(module, global_table, cast<Constant>(CE->getOperand(1)));
            return lhs ^ rhs;
        } else if(CE->getOpcode() == Instruction::Select) {
            auto pred = resolveConstantInteger(module, global_table, cast<Constant>(CE->getOperand(0)));
            auto trueVal = resolveConstantInteger(module, global_table, cast<Constant>(CE->getOperand(1)));
            auto falseVal = resolveConstantInteger(module, global_table, cast<Constant>(CE->getOperand(2)));
            return pred ? trueVal : falseVal;
        } else if(CE->isGEPWithNoNotionalOverIndexing()) {
            auto base = resolveConstantInteger(module, global_table, cast<Constant>(CE->getOperand(0)));
            SmallVector<Value *, 8> indices(CE->op_begin()+1, CE->op_end());
            auto offset = module->getDataLayout().getIndexedOffset(
                CE->getOperand(0)->getType(),
                indices);
            return base + offset;
        } else {
            errs() << "Unsupported constant expression " << *value << "\n";
            report_fatal_error("Unsupported constant expression");
            abort();
        }
    }
    if(auto GO = dyn_cast<GlobalObject>(value)) {
        return global_table[GO];
    }
    if(isa<ConstantPointerNull>(value)) {
        return 0;
    }
    if(auto CI = dyn_cast<ConstantInt>(value)) {
        return CI->getZExtValue();
    }

    errs() << "Unsupported constant " << *value << "\n";
    report_fatal_error("Unsupported constant");
    abort();
}

static std::unordered_map<Constant *, uint64_t> constant_value_cache;

static uint64_t resolveConstantInteger(std::unique_ptr<Module> &module, std::unordered_map<GlobalObject *, unsigned long long> &global_table, Constant *value) {
    auto itr = constant_value_cache.find(value);
    if(itr != constant_value_cache.end()) {
        return itr->second;
    }
    auto integer = reallyResolveConstantInteger(module, global_table, value);
    constant_value_cache[value] = integer;
    return integer;
}

static std::string reallyValueForPrinting(std::unique_ptr<Module> &module,
                                          std::unordered_map<GlobalObject *, unsigned long long> &global_table,
                                          Constant *value) {
    if(auto cint = dyn_cast<ConstantInt>(value)) {
        return cint->getValue().toString(10, false);
    }
    if(isa<UndefValue>(value)) {
        auto ty = value->getType();
        if(ty->isIntegerTy() || ty->isPointerTy()) {
            return "0 #|undef|#";
        } else if(ty->isFloatTy()) {
            return "0.0f0 #|undef|#";
        } else if(ty->isDoubleTy()) {
            return "0.0d0 #|undef|#";
        }
    }
    if(isa<ConstantPointerNull>(value)) {
        return "0 #|ptr-null|#";
    }
    if(auto global = dyn_cast<GlobalObject>(value)) {
        auto itr = global_table.find(global);
        if(itr != global_table.end()) {
            return std::to_string(itr->second);
        }
    }

    if(auto expr = dyn_cast<ConstantExpr>(value)) {
        return std::to_string(resolveConstantInteger(module, global_table, expr));
    }

    if(auto CFP = dyn_cast<ConstantFP>(value)) {
        SmallVector<char, 32> buf;
        CFP->getValueAPF().toString(buf);
        for(auto &ch: buf) {
            if(ch == 'E') {
                if(CFP->getType()->isFloatTy()) {
                    ch = 'f';
                } else if(CFP->getType()->isDoubleTy() || CFP->getType()->isX86_FP80Ty()) {
                    ch = 'd';
                }
            }
        }
        std::string str(buf.begin(), buf.end());
        if(CFP->getType()->isFloatTy()) {
            return "(float " + str + ")";
        } else if(CFP->getType()->isDoubleTy() || CFP->getType()->isX86_FP80Ty()) {
            return "(float " + str + " 0.0d0)";
        }
    }

    errs() << "Unable to print constant " << *value << "\n";
    report_fatal_error("Unable to print constant");
    abort();
}

static std::unordered_map<Constant *, std::string> constant_representation_cache;

static StringRef valueForPrinting(std::unique_ptr<Module> &module,
                                  std::unordered_map<Value*, std::string> &name_table,
                                  std::unordered_map<GlobalObject *, unsigned long long> &global_table,
                                  Value *value) {
    if(isa<Instruction>(value) || isa<Argument>(value)) {
        return name_table[value];
    }
    if(auto C = dyn_cast<Constant>(value)) {
        auto itr = constant_representation_cache.find(C);
        if(itr != constant_representation_cache.end()) {
            return itr->second;
        }
        auto repr = reallyValueForPrinting(module, global_table, C);
        constant_representation_cache[C] = repr;
        return constant_representation_cache[C];
    }

    errs() << "Unable to print value " << *value << "\n";
    report_fatal_error("Unable to print value");
    abort();
}

static void prepareForBranch(std::unique_ptr<Module> &module,
                             std::unordered_map<Value*, std::string> &name_table,
                             std::unordered_map<GlobalObject *, unsigned long long> &global_table,
                             BasicBlock *origin, BasicBlock *destination, const char *indent) {
    bool first = true;
    for(auto &inst: *destination) {
        auto phi = dyn_cast<PHINode>(&inst);
        if(!phi) {
            break;
        }
        if(first) {
            outs() << indent << "(psetq ";
            first = false;
        } else {
            outs() << "\n" << indent << "       ";
        }

        auto val = phi->getIncomingValueForBlock(origin);
        outs() << name_table[&inst] << " " << valueForPrinting(module, name_table, global_table, val);
    }
    if(!first) {
        outs() << ")\n";
    }
}

static std::string reallyTypeSuffix(std::unique_ptr<Module> &module, Type *ty) {
    if(auto ity = dyn_cast<IntegerType>(ty)) {
        return "i" + std::to_string(ity->getBitWidth());
    }
    if(ty->isFloatTy()) {
        return "f32";
    }
    if(ty->isDoubleTy() || ty->isX86_FP80Ty()) { // FIXME: Should eliminate fp80...
        return "f64";
    }
    if(auto PTy = dyn_cast<PointerType>(ty)) {
        return "i" + std::to_string(module->getDataLayout().getPointerSizeInBits(PTy->getAddressSpace()));
    }
    errs() << "Unsupported type " << *ty << "\n";
    report_fatal_error("Unsupported type suffix");
    abort();
}

static std::unordered_map<Type *, std::string> type_suffix_cache;

static StringRef typeSuffix(std::unique_ptr<Module> &module, Type *ty) {
    auto itr = type_suffix_cache.find(ty);
    if(itr != type_suffix_cache.end()) {
        return itr->second;
    }

    auto value = reallyTypeSuffix(module, ty);
    type_suffix_cache[ty] = value;
    return type_suffix_cache[ty];
}

static const char *getPredicateText(unsigned predicate) {
  const char * pred = "unknown";
  switch (predicate) {
  case FCmpInst::FCMP_FALSE: pred = "false"; break;
  case FCmpInst::FCMP_OEQ:   pred = "oeq"; break;
  case FCmpInst::FCMP_OGT:   pred = "ogt"; break;
  case FCmpInst::FCMP_OGE:   pred = "oge"; break;
  case FCmpInst::FCMP_OLT:   pred = "olt"; break;
  case FCmpInst::FCMP_OLE:   pred = "ole"; break;
  case FCmpInst::FCMP_ONE:   pred = "one"; break;
  case FCmpInst::FCMP_ORD:   pred = "ord"; break;
  case FCmpInst::FCMP_UNO:   pred = "uno"; break;
  case FCmpInst::FCMP_UEQ:   pred = "ueq"; break;
  case FCmpInst::FCMP_UGT:   pred = "ugt"; break;
  case FCmpInst::FCMP_UGE:   pred = "uge"; break;
  case FCmpInst::FCMP_ULT:   pred = "ult"; break;
  case FCmpInst::FCMP_ULE:   pred = "ule"; break;
  case FCmpInst::FCMP_UNE:   pred = "une"; break;
  case FCmpInst::FCMP_TRUE:  pred = "true"; break;
  case ICmpInst::ICMP_EQ:    pred = "eq"; break;
  case ICmpInst::ICMP_NE:    pred = "ne"; break;
  case ICmpInst::ICMP_SGT:   pred = "sgt"; break;
  case ICmpInst::ICMP_SGE:   pred = "sge"; break;
  case ICmpInst::ICMP_SLT:   pred = "slt"; break;
  case ICmpInst::ICMP_SLE:   pred = "sle"; break;
  case ICmpInst::ICMP_UGT:   pred = "ugt"; break;
  case ICmpInst::ICMP_UGE:   pred = "uge"; break;
  case ICmpInst::ICMP_ULT:   pred = "ult"; break;
  case ICmpInst::ICMP_ULE:   pred = "ule"; break;
  }
  return pred;
}

static std::string reallyClTypeName(Type *ty) {
    if(ty->isPointerTy()) {
        return "(unsigned-byte 64)";
    } else if(ty->isIntegerTy()) {
        return "(unsigned-byte " + std::to_string(ty->getIntegerBitWidth()) + ")";
    } else if(ty->isFloatTy()) {
        return "single-float";
    } else if(ty->isDoubleTy() || ty->isX86_FP80Ty()) { // FIXME: Should eliminate fp80...
        return "double-float";
    } else {
        errs() << "Unsupported instruction type " << *ty << "\n";
        report_fatal_error("Unsupported instruction type ");
    }
}

static std::unordered_map<Type *, std::string> cl_type_name_cache;

static StringRef clTypeName(Type *ty) {
    auto itr = cl_type_name_cache.find(ty);
    if(itr != cl_type_name_cache.end()) {
        return itr->second;
    }
    auto value = reallyClTypeName(ty);
    cl_type_name_cache[ty] = value;
    return cl_type_name_cache[ty];
}

static const char *clTypeInitializer(Type *ty) {;
    if(ty->isIntegerTy() || ty->isPointerTy()) {
        return "0";
    } else if(ty->isFloatTy()) {
        return "0.0f0";
    } else if(ty->isDoubleTy() || ty->isX86_FP80Ty()) { // FIXME: Should eliminate fp80...
        return "0.0d0";
    } else {
        errs() << "Unsupported instruction type " << *ty << "\n";
        report_fatal_error("Unsupported instruction type");
    }
}

static Instruction *nextInstruction(Instruction *inst) {
    BasicBlock::iterator I(inst);
    ++I;
    return &*I;
}

static bool isFusableCmpAndBranchPair(Instruction &inst) {
    // A cmp/br pair can be fused if the branch's condition is the comparison
    // and the comparison is only used by the branch.
    if(!isa<CmpInst>(inst)) {
        return false;
    }
    auto next = nextInstruction(&inst);
    auto B = dyn_cast<BranchInst>(next);
    if(!B) {
        return false;
    }
    if(!B->isConditional()) {
        return false;
    }
    if(B->getCondition() != &inst) {
        return false;
    }
    if(!inst.hasOneUse()) {
        return false;
    }
    return true;
}

static bool isSetjmpCall(Instruction &inst) {
    auto CI = dyn_cast<CallInst>(&inst);
    if(!CI) {
        return false;
    }
    if(!CI->canReturnTwice()) {
        return false;
    }
    return true;
}

static void translateFunction(std::unique_ptr<Module> &module, std::unordered_map<GlobalObject *, unsigned long long> &global_table, Function &fn) {
    auto &DL = module->getDataLayout();
    outs() << "(define-llvm-function " << escapeSymbol(fn.getName()) << " ((";
    std::unordered_map<Value*, std::string> name_table;
    for(auto &arg: fn.args()) {
        std::string name = arg.getName();
        if(name.empty()) {
            name += "%";
        } else {
            name += ".";
        }
        name += std::to_string(name_table.size());
        name_table[&arg] = escapeSymbol(name);
        outs() << escapeSymbol(name) << " ";
    }
    outs() << ")";
    bool need_frame_pointer = false;
    for(auto &inst: fn.getEntryBlock()) {
        if(isa<AllocaInst>(inst)) {
            need_frame_pointer = true;
            break;
        }
    }
    outs() << " :need-frame-pointer " << (need_frame_pointer ? "t" : "nil");
    if(fn.callsFunctionThatReturnsTwice()) {
        outs() << " :uses-setjmp t";
    }
    outs() << ")\n";
    if(!OptimizeQualities.empty()) {
        outs() << "  (declare (optimize " << OptimizeQualities << "))\n";
    }
    // Create names for each basic block.
    for(auto &bb: fn) {
        std::string name = bb.getName();
        if(name.empty()) {
            name += "%";
        } else {
            name += ".";
        }
        name += std::to_string(name_table.size());
        name_table[&bb] = escapeSymbol(name);
    }
    // Create variables for each instruction.
    outs() << "  (let (";
    for(auto &bb: fn) {
        for(auto &inst: bb) {
            auto ty = inst.getType();
            if(ty->isVoidTy()) {
                continue;
            }
            if(inst.use_empty()) {
                continue;
            }
            if(isFusableCmpAndBranchPair(inst)) {
                continue;
            }
            std::string name = inst.getName();
            if(name.empty()) {
                name += "%";
            } else {
                name += ".";
            }
            name += std::to_string(name_table.size());
            name_table[&inst] = escapeSymbol(name);
            outs() << "(" << escapeSymbol(name) << " " << clTypeInitializer(ty) << ")\n        ";
            if(isSetjmpCall(inst)) {
                // Generate a setjmp thunk variable for this instruction.
                outs() << "(setjmp-thunk." << escapeSymbol(name) << " nil)\n        ";
            }
        }
    }
    outs() << ")\n";
    if(EmitTypeDeclarartions) {
        outs() << "    (declare ";
        for(auto &arg: fn.args()) {
            auto ty = arg.getType();
            outs() << "(type " << clTypeName(ty) << " " << name_table[&arg] << ")\n             ";
        }
        for(auto &bb: fn) {
            for(auto &inst: bb) {
                auto ty = inst.getType();
                if(ty->isVoidTy()) {
                    continue;
                }
                if(inst.use_empty()) {
                    continue;
                }
                if(isFusableCmpAndBranchPair(inst)) {
                    continue;
                }
                outs() << "(type " << clTypeName(ty) << " " << name_table[&inst] << ")\n             ";
            }
        }
        outs() << ")\n";
    }
    outs() << "    (block nil\n";
    outs() << "      (tagbody\n";
    for(auto &bb: fn) {
        for(auto &inst: bb) {
            if(isSetjmpCall(inst)) {
                // Generate a setjmp thunk variable for this instruction.
                outs() << "        (setf setjmp-thunk." << name_table[&inst] << " (setjmp.prepare setjmp-target." << name_table[&inst] << "))\n";
            }
        }
    }
    for(auto &bb: fn) {
        outs() << "       " << name_table[&bb] << "\n";
        bool skip_next = false;
        for(auto &inst: bb) {
            if(PrintLLVMInstructions) {
                outs() << "        #| " << inst << "|#\n";
            }
            if(skip_next) {
                skip_next = false;
                continue;
            }
            if(inst.getType()->isVoidTy() ||
               isFusableCmpAndBranchPair(inst) ||
               inst.use_empty() ||
               isSetjmpCall(inst)) {
                outs() << "        ";
            } else if(isa<PHINode>(inst)) {
            } else {
                outs() << "        (setq " << name_table[&inst] << " ";
            }
            switch(inst.getOpcode()) {
            case Instruction::Alloca: {
                AllocaInst &AI = cast<AllocaInst>(inst);
                if(!AI.isStaticAlloca()) {
                    report_fatal_error("Non-static alloca");
                }
                auto size = cast<ConstantInt>(AI.getArraySize());
                outs() << "(alloca " << (size->getZExtValue() * DL.getTypeAllocSize(AI.getAllocatedType())) << ")";
                break;
            }
                // These are no-op casts.
            case Instruction::PtrToInt:
            case Instruction::IntToPtr: {
                outs() << valueForPrinting(module, name_table, global_table, inst.getOperand(0));
                break;
            }
            case Instruction::Add:
            case Instruction::FAdd:
            case Instruction::Sub:
            case Instruction::FSub:
            case Instruction::Mul:
            case Instruction::FMul:
            case Instruction::UDiv:
            case Instruction::SDiv:
            case Instruction::FDiv:
            case Instruction::URem:
            case Instruction::SRem:
            case Instruction::FRem:
            case Instruction::Shl:
            case Instruction::LShr:
            case Instruction::AShr:
            case Instruction::And:
            case Instruction::Or:
            case Instruction::Xor: {
                outs() << "(" << inst.getOpcodeName(inst.getOpcode()) << "." << typeSuffix(module, inst.getType())
                       << " " << valueForPrinting(module, name_table, global_table, inst.getOperand(0))
                       << " " << valueForPrinting(module, name_table, global_table, inst.getOperand(1)) << ")";
                break;
            }
            case Instruction::BitCast:
            case Instruction::Trunc:
            case Instruction::ZExt:
            case Instruction::SExt:
            case Instruction::FPToUI:
            case Instruction::FPToSI:
            case Instruction::UIToFP:
            case Instruction::SIToFP:
            case Instruction::FPTrunc:
            case Instruction::FPExt: {
                outs() << "(" << inst.getOpcodeName(inst.getOpcode()) << "." << typeSuffix(module, inst.getOperand(0)->getType()) << "." << typeSuffix(module, inst.getType())
                       << " " << valueForPrinting(module, name_table, global_table, inst.getOperand(0)) << ")";
                break;
            }
            case Instruction::ICmp:
            case Instruction::FCmp: {
                auto &CI = cast<CmpInst>(inst);
                if(isFusableCmpAndBranchPair(inst)) {
                    skip_next = true;
                    auto B = cast<BranchInst>(nextInstruction(&CI));
                    outs() << "(if (" << inst.getOpcodeName(CI.getOpcode()) << "." << getPredicateText(CI.getPredicate()) << ".fused" << "." << typeSuffix(module, CI.getOperand(0)->getType())
                           << " " << valueForPrinting(module, name_table, global_table, CI.getOperand(0))
                           << " " << valueForPrinting(module, name_table, global_table, CI.getOperand(1)) << ")\n";
                    outs() << "            (progn\n";
                    prepareForBranch(module, name_table, global_table, &bb, B->getSuccessor(0), "              ");
                    outs() << "              (go " << name_table[B->getSuccessor(0)] << "))\n";
                    outs() << "            (progn\n";
                    prepareForBranch(module, name_table, global_table, &bb, B->getSuccessor(1), "              ");
                    outs() << "              (go " << name_table[B->getSuccessor(1)] << ")))";
                } else {
                    outs() << "(" << inst.getOpcodeName(CI.getOpcode()) << "." << getPredicateText(CI.getPredicate()) << "." << typeSuffix(module, CI.getOperand(0)->getType())
                           << " " << valueForPrinting(module, name_table, global_table, CI.getOperand(0))
                           << " " << valueForPrinting(module, name_table, global_table, CI.getOperand(1)) << ")";
                }
                break;
            }
            case Instruction::Load:
                outs() << "(load." << typeSuffix(module, inst.getType())
                       << " " << valueForPrinting(module, name_table, global_table, inst.getOperand(0)) << ")";
                break;
            case Instruction::Store:
                outs() << "(store." << typeSuffix(module, inst.getOperand(0)->getType())
                       << " " << valueForPrinting(module, name_table, global_table, inst.getOperand(0))
                       << " " << valueForPrinting(module, name_table, global_table, inst.getOperand(1)) << ")";
                break;
            case Instruction::PHI:
                break;
            case Instruction::Br: {
                auto &B = cast<BranchInst>(inst);
                if(B.isConditional()) {
                    outs() << "(if (not (eql " << valueForPrinting(module, name_table, global_table, B.getCondition()) << " 0))\n";
                    outs() << "            (progn\n";
                    prepareForBranch(module, name_table, global_table, &bb, B.getSuccessor(0), "              ");
                    outs() << "              (go " << name_table[B.getSuccessor(0)] << "))\n";
                    outs() << "            (progn\n";
                    prepareForBranch(module, name_table, global_table, &bb, B.getSuccessor(1), "              ");
                    outs() << "              (go " << name_table[B.getSuccessor(1)] << ")))";
                } else {
                    prepareForBranch(module, name_table, global_table, &bb, B.getSuccessor(0), "");
                    if(isa<PHINode>(B.getSuccessor(0)->front())) {
                        outs() << "        ";
                    }
                    outs() << "(go " << name_table[B.getSuccessor(0)] << ")";
                }
                break;
            }
            case Instruction::Ret: {
                auto &ret = cast<ReturnInst>(inst);
                auto val = ret.getReturnValue();
                if(val) {
                    outs() << "(return " << valueForPrinting(module, name_table, global_table, val) << ")";
                } else {
                    outs() << "(return)";
                }
                break;
            }
            case Instruction::Unreachable: {
                outs() << "(error \"Reached unreachable!\")";
                break;
            }
            case Instruction::Call: {
                auto &call = cast<CallInst>(inst);
                if(call.isInlineAsm()) {
                    errs() << "Unsupported instruction " << inst << "\n";
                    report_fatal_error("Inline asm? You must be joking!");
                }
                if(isSetjmpCall(inst)) {
                    outs() << "(call-direct " << escapeSymbol(call.getCalledFunction()->getName()) << " setjmp-thunk." << name_table[&inst];
                    for(auto &operand: call.arg_operands()) {
                        outs() << " " << valueForPrinting(module, name_table, global_table, operand);
                    }
                    outs() << ")\n";
                    outs() << "        (setq " << name_table[&inst] << " 0)\n";
                    outs() << "        (go setjmp-resume." << name_table[&inst] << ")\n";
                    outs() << "      setjmp-target." << name_table[&inst] << "\n";
                    outs() << "        (setq " << name_table[&inst] << " 1)\n";
                    outs() << "        (go setjmp-resume." << name_table[&inst] << ")\n";
                    outs() << "      setjmp-resume." << name_table[&inst] << "\n";
                } else if(auto target = call.getCalledFunction()) {
                    outs() << "(call-direct " << escapeSymbol(target->getName());
                    for(auto &operand: call.arg_operands()) {
                        outs() << " " << valueForPrinting(module, name_table, global_table, operand);
                    }
                    outs() << ")";
                } else {
                    outs() << "(call-indirect " << valueForPrinting(module, name_table, global_table, call.getCalledValue());
                    for(auto &operand: call.arg_operands()) {
                        outs() << " " << valueForPrinting(module, name_table, global_table, operand);
                    }
                    outs() << ")";
                }
                break;
            }
            case Instruction::Select: {
                auto &select = cast<SelectInst>(inst);
                outs() << "(select " << valueForPrinting(module, name_table, global_table, select.getCondition());
                outs() << " " << valueForPrinting(module, name_table, global_table, select.getTrueValue());
                outs() << " " << valueForPrinting(module, name_table, global_table, select.getFalseValue()) << ")";
                break;
            }
            case Instruction::GetElementPtr: {
                auto &GEP = cast<GetElementPtrInst>(inst);
                if(!GEP.hasAllConstantIndices()) {
                    errs() << "Unsupported instruction " << inst << "\n";
                    report_fatal_error(std::string("Unsupported instruction ") +
                                       inst.getOpcodeName(inst.getOpcode()) + " (with non-constant indicies)");
                }
                APInt offset(DL.getPointerSizeInBits(GEP.getPointerAddressSpace()), 0);
                if(!GEP.accumulateConstantOffset(DL, offset)) {
                    report_fatal_error("Unable to compute GEP offset.");
                }
                outs() << "(add." << typeSuffix(module, GEP.getType())
                       << " " << valueForPrinting(module, name_table, global_table, GEP.getPointerOperand())
                       << " " << offset << ")";
                break;
            }
            default:
                errs() << "Unsupported instruction " << inst << "\n";
                report_fatal_error(std::string("Unsupported instruction ") +
                                   inst.getOpcodeName(inst.getOpcode()));
            }
            if(inst.getType()->isVoidTy() ||
               isFusableCmpAndBranchPair(inst) ||
               inst.use_empty() ||
               isSetjmpCall(inst)) {
                outs() << "\n";
            } else if(isa<PHINode>(inst)) {
            } else {
                outs() << ")\n";
            }
        }
    }
    outs() << "))))\n\n";
}

static void storeInteger(std::vector<uint8_t> &data_section, uint64_t offset, uint64_t value, int width) {
    switch(width) {
    case 64:
        data_section[offset+7] = value >> 56;
        data_section[offset+6] = value >> 48;
        data_section[offset+5] = value >> 40;
        data_section[offset+4] = value >> 32;
    case 32:
        data_section[offset+3] = value >> 24;
        data_section[offset+2] = value >> 16;
    case 16:
        data_section[offset+1] = value >> 8;
    case 8:
        data_section[offset+0] = value;
        break;
    default:
        report_fatal_error("Cannot store integer constant of unusual width.");
    }
}

static void storeGlobalConstant(std::unique_ptr<Module> &module, std::unordered_map<GlobalObject *, unsigned long long> &global_table, std::vector<uint8_t> &data_section, uint64_t offset, Constant *value) {
    if(auto CA = dyn_cast<ConstantArray>(value)) {
        auto ty = CA->getType();
        auto inner_ty = ty->getElementType();
        auto elt_size = module->getDataLayout().getTypeAllocSize(inner_ty);
        for(unsigned i = 0; i < ty->getNumElements(); i += 1) {
            storeGlobalConstant(module, global_table, data_section, offset + i * elt_size, CA->getOperand(i));
        }
    } else if(auto CDA = dyn_cast<ConstantDataArray>(value)) {
        auto ty = CDA->getType();
        auto inner_ty = ty->getElementType();
        auto elt_size = module->getDataLayout().getTypeAllocSize(inner_ty);
        for(unsigned i = 0; i < ty->getNumElements(); i += 1) {
            storeGlobalConstant(module, global_table, data_section, offset + i * elt_size, CDA->getElementAsConstant(i));
        }
    } else if(auto CS = dyn_cast<ConstantStruct>(value)) {
        auto layout = module->getDataLayout().getStructLayout(CS->getType());
        for(unsigned i = 0; i < CS->getNumOperands(); i += 1) {
            storeGlobalConstant(module, global_table, data_section, offset + layout->getElementOffset(i), cast<Constant>(CS->getOperand(i)));
        }
    } else if(auto CFP = dyn_cast<ConstantFP>(value)) {
        if(CFP->getType()->isFloatTy()) {
            storeInteger(data_section, offset, CFP->getValueAPF().bitcastToAPInt().getZExtValue(), 32);
        } else if(CFP->getType()->isDoubleTy()) {
            storeInteger(data_section, offset, CFP->getValueAPF().bitcastToAPInt().getZExtValue(), 64);
        } else {
            report_fatal_error("Unsupported float type");
        }
    } else if(isa<ConstantAggregateZero>(value) || isa<UndefValue>(value)) {
        /* Nuthin' at all */
    } else {
        uint64_t numeric_value = resolveConstantInteger(module, global_table, value);
        storeInteger(data_section, offset, numeric_value,
                     module->getDataLayout().getTypeStoreSizeInBits(value->getType()));
    }
}

int main(int argc, char **argv) {
    sys::PrintStackTraceOnErrorSignal();
    PrettyStackTraceProgram X(argc, argv);

    llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

    SmallVector<const char *, 32> modified_args(&argv[0], &argv[argc]);
    modified_args.push_back("-scalarize-load-store");

    PassRegistry &Registry = *PassRegistry::getPassRegistry();
    initializeCore(Registry);
    initializeScalarOpts(Registry);

    cl::ParseCommandLineOptions(modified_args.size(), &modified_args[0],
                                "Iota LLVM -> CL transpiler\n");

    LLVMContext context;
    SMDiagnostic err;
    auto module = parseIRFile(InputFilename, err, context);

    if(!module) {
        err.print("iota", errs());
        return -1;
    }

    if(!TranslateOnly) {
        lowerVectorBswap(*module);

        legacy::PassManager PM;

        // Internalize all symbols in the module except the entry point.
        const char *export_name = EntryPoint.c_str();
        PM.add(createInternalizePass(export_name));

        // Simplification passes.
        PM.add(createExpandVarArgsPass());
        PM.add(createScalarizerPass());
        PM.add(createSimplifyAllocasPass());
        PM.add(createExpandGetElementPtrPass());
        PM.add(createExpandArithWithOverflowPass());
        PM.add(createLowerSwitchPass());
        PM.add(createPromoteI1OpsPass());
        PM.add(createPromoteIntegersPass());
        PM.add(createExpandLargeIntegersPass());

        PM.run(*module);

        lowerBitcastTrunc(*module);
        eliminateLifetimeIntrinsics(*module);

        legacy::PassManager PM2;
        PM2.add(createInstructionCombiningPass());
        PM2.add(createStripSymbolsPass(true));
        PM2.add(createGlobalDCEPass());
        PM2.run(*module);
    }

    unsigned long long data_origin = 0x200000;
    unsigned long long data_end = data_origin;
    unsigned long long next_fn_id = 1;
    // Lay out the data section.
    std::unordered_map<GlobalObject *, unsigned long long> global_table;
    for(auto &glob: module->globals()) {
        if(!glob.hasInitializer()) {
            report_fatal_error("Global " + glob.getName() + " is uninitialized");
        }
        global_table[&glob] = data_end;
        data_end += module->getDataLayout().getTypeAllocSize(glob.getInitializer()->getType());
        // Round to 16 byte boundary.
        data_end += 15;
        data_end &= ~15ull;
    }
    // Lay out the function table.
    std::vector<Function *> function_table;
    for(auto &fn: module->functions()) {
        if(!fn.hasAddressTaken()) {
            continue;
        }
        function_table.push_back(&fn);
        global_table[&fn] = next_fn_id;
        next_fn_id += 1;
    }
    // Initialize the data section.
    std::vector<uint8_t> data_section;
    data_section.resize(data_end - data_origin, 0);
    for(auto &glob: module->globals()) {
        auto offset = global_table[&glob] - data_origin;
        storeGlobalConstant(module, global_table, data_section, offset, glob.getInitializer());
    }

    if(!PackageName.empty()) {
        outs() << "(in-package " << PackageName << ")\n\n";
    }

    outs() << "(defun make-context (&rest personality-initargs)\n";
    outs() << "  (make-llvm-context\n";
    if(ContextPersonality.empty()) {
        outs() << "   :unix\n";
    } else {
        outs() << "   '" << ContextPersonality << "\n";
    }
    outs() << "   " << data_origin << "\n";
    outs() << "   #.(make-array " << (data_end - data_origin) << " :element-type '(unsigned-byte 8) :initial-contents '(";
    for(size_t i = 0; i < data_section.size(); i += 1) {
        if((i % 64) == 0) {
            outs() << "\n     ";
        } else {
            outs() << " ";
        }
        outs() << (int)data_section[i];
    }
    outs() << "))\n";
    outs() << "   " << (module->getDataLayout().getPointerSizeInBits(0) == 32 ? "t" : "nil") << "\n";
    outs() << "   #(";
    for(size_t i = 0; i < function_table.size(); i += 1) {
        if((i % 64) == 0) {
            outs() << "\n     ";
        } else {
            outs() << " ";
        }
        outs() << escapeSymbol(function_table[i]->getName());
    }
    outs() << ")\n";
    outs() << "   '" << escapeSymbol(EntryPoint) << "\n";
    outs() << "   personality-initargs))\n\n";

    errs() << "Undefined functions:\n";
    for(auto &fn: module->functions()) {
        if(fn.empty()) {
            errs() << fn.getName() << "\n";
        }
    }

    unsigned functions_to_translate = 0;
    for(auto &fn: module->functions()) {
        if(!fn.empty() && !fn.isDefTriviallyDead()) {
            functions_to_translate += 1;
        }
    }
    errs() << functions_to_translate << " functions to translate\n";

    // Translate all functions.
    unsigned functions_translated = 0;
    for(auto &fn: module->functions()) {
        if(fn.isVarArg()) {
            report_fatal_error("Function " + fn.getName() + " is vararg");
        }
        if(!fn.empty() && !fn.isDefTriviallyDead()) {
            errs() << functions_translated << ": Translating function " << fn.getName() << "\n";
            functions_translated += 1;
            translateFunction(module, global_table, fn);
        }
    }
}
