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

#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/Transforms/NaCl.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include <unordered_map>

using namespace llvm;

extern void lowerBitcastTrunc(Module &M);
extern void lowerVectorBswap(Module &M);
extern void eliminateLifetimeIntrinsics(Module &M);

static cl::list<std::string>
InputFilenames(cl::Positional, cl::desc("<input files>"), cl::OneOrMore);

static cl::list<std::string>
LibrarySearchPaths("L", cl::desc("Library search paths"), cl::ZeroOrMore, cl::Prefix);

static cl::list<std::string>
InputLibraries("l", cl::desc("Libraries to link"), cl::ZeroOrMore, cl::Prefix);

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

static cl::opt<bool>
LibraryMode("library-mode", cl::desc("Don't export _start, only symbols specified with -internalize-public-api-file or -internalize-public-api-list"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"),
               cl::value_desc("filename"));

class IotaTranslator {
public:
    Module *module;
    std::unordered_map<GlobalObject *, unsigned long long> global_table;
};

class IotaFunctionTranslator {
public:
    IotaTranslator *global;
    std::unordered_map<Value*, std::string> name_table;
    DominatorTree dom;

    StringRef valueForPrinting(Value *value);
    void translateBlock(BasicBlock &bb, std::unique_ptr<tool_output_file> &Out);
};

class IotaBlockTranslator {
public:
    IotaFunctionTranslator *function;
    std::vector<Instruction *> remaining;

    std::string translateInstruction(Instruction *inst);
    bool maybeFuseValue(Instruction *user, Value *value);
    std::string useValue(Instruction *user, Value *value);
    std::string prepareForBranch(Instruction *user, BasicBlock *destination, const char *indent, bool use_value);
};

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

static uint64_t resolveConstantInteger(Module *module, std::unordered_map<GlobalObject *, unsigned long long> &global_table, Constant *value);

static uint64_t reallyResolveConstantInteger(Module *module, std::unordered_map<GlobalObject *, unsigned long long> &global_table, Constant *value) {
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

static uint64_t resolveConstantInteger(Module *module, std::unordered_map<GlobalObject *, unsigned long long> &global_table, Constant *value) {
    auto itr = constant_value_cache.find(value);
    if(itr != constant_value_cache.end()) {
        return itr->second;
    }
    auto integer = reallyResolveConstantInteger(module, global_table, value);
    constant_value_cache[value] = integer;
    return integer;
}

static std::string reallyValueForPrinting(Module *module,
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
        // FIXME: These should generate proper CL floats with appropriate exponent characters.
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

StringRef IotaFunctionTranslator::valueForPrinting(Value *value) {
    if(isa<Instruction>(value) || isa<Argument>(value)) {
        return name_table[value];
    }
    if(auto C = dyn_cast<Constant>(value)) {
        auto itr = constant_representation_cache.find(C);
        if(itr != constant_representation_cache.end()) {
            return itr->second;
        }
        auto repr = reallyValueForPrinting(global->module, global->global_table, C);
        constant_representation_cache[C] = repr;
        return constant_representation_cache[C];
    }

    errs() << "Unable to print value " << *value << "\n";
    report_fatal_error("Unable to print value");
    abort();
}

std::string IotaBlockTranslator::prepareForBranch(Instruction *user, BasicBlock *destination, const char *indent, bool use_value) {
    BasicBlock *origin = user->getParent();
    std::string result;
    bool first = true;
    for(auto &inst: *destination) {
        auto phi = dyn_cast<PHINode>(&inst);
        if(!phi) {
            break;
        }
        if(first) {
            result += indent + std::string("(psetq ");
            first = false;
        } else {
            result += std::string("\n") + indent + "       ";
        }

        auto val = phi->getIncomingValueForBlock(origin);
        result += function->name_table[&inst] + " ";
        if(use_value) {
            result += useValue(user, val);
        } else {
            result += function->valueForPrinting(val);
        }
    }
    if(!first) {
        result += ")\n";
    }
    return result;
}

static std::string reallyTypeSuffix(Module *module, Type *ty) {
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

static StringRef typeSuffix(Module *module, Type *ty) {
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

static std::string reallyClTypeName(Module *module, Type *ty) {
    if(ty->isPointerTy()) {
        return "(unsigned-byte " + std::to_string(module->getDataLayout().getPointerSizeInBits(ty->getPointerAddressSpace())) + ")";
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

static StringRef clTypeName(Module *module, Type *ty) {
    auto itr = cl_type_name_cache.find(ty);
    if(itr != cl_type_name_cache.end()) {
        return itr->second;
    }
    auto value = reallyClTypeName(module, ty);
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

bool IotaBlockTranslator::maybeFuseValue(Instruction *user, Value *value) {
    if(isa<Instruction>(value) &&
       value->hasOneUse() &&
       !isSetjmpCall(cast<Instruction>(*value)) &&
       !remaining.empty() && remaining.back() == value) {
        // if the last instruction on the stack is this value and it
        // only has one use, then fold that instruction into this one.
        remaining.pop_back();
        return true;
    } else {
        return false;
    }
}

std::string IotaBlockTranslator::useValue(Instruction *user, Value *value) {
    if(maybeFuseValue(user, value)) {
        return translateInstruction(cast<Instruction>(value));
    }

    return this->function->valueForPrinting(value);
}

std::string IotaBlockTranslator::translateInstruction(Instruction *inst) {
    switch(inst->getOpcode()) {
    case Instruction::Alloca: {
        auto &DL = function->global->module->getDataLayout();
        AllocaInst &AI = cast<AllocaInst>(*inst);
        if(!AI.isStaticAlloca()) {
            report_fatal_error("Non-static alloca");
        }
        auto size = cast<ConstantInt>(AI.getArraySize());
        return std::string("(alloca ") + std::to_string(size->getZExtValue() * DL.getTypeAllocSize(AI.getAllocatedType())) + ")";
    }
    case Instruction::PtrToInt:
    case Instruction::IntToPtr: {
        // These are no-op casts.
        return useValue(inst, inst->getOperand(0));
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
        // Proper evaluation order...
        auto rhs = useValue(inst, inst->getOperand(1));
        auto lhs = useValue(inst, inst->getOperand(0));
        return std::string("(") + std::string(inst->getOpcodeName(inst->getOpcode())) + "." + std::string(typeSuffix(function->global->module, inst->getType()))
            + " " + lhs + " " + rhs + ")";
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
        auto lhs_type = typeSuffix(function->global->module, inst->getOperand(0)->getType());
        auto rhs_type = typeSuffix(function->global->module, inst->getType());
        // If this is a bitcast and both side have the same type, then just return the operand.
        if(inst->getOpcode() == Instruction::BitCast && lhs_type == rhs_type) {
            return useValue(inst, inst->getOperand(0));
        } else {
            return std::string("(") + std::string(inst->getOpcodeName(inst->getOpcode())) + "." + std::string(lhs_type) + "." + std::string(rhs_type)
                + " " + useValue(inst, inst->getOperand(0)) + ")";
        }
    }
    case Instruction::ICmp:
    case Instruction::FCmp: {
        auto &CI = cast<CmpInst>(*inst);
        // Proper evaluation order...
        auto rhs = useValue(&CI, CI.getOperand(1));
        auto lhs = useValue(&CI, CI.getOperand(0));
        return std::string("(") + std::string(inst->getOpcodeName(CI.getOpcode())) + "." + getPredicateText(CI.getPredicate()) + "." + std::string(typeSuffix(function->global->module, CI.getOperand(0)->getType()))
               + " " + lhs + " " + rhs + ")";
    }
    case Instruction::Load: {
        return std::string("(load.") + std::string(typeSuffix(function->global->module, inst->getType())) + " " + useValue(inst, inst->getOperand(0)) + ")";
    }
    case Instruction::Store: {
        // Proper evaluation order...
        auto rhs = useValue(inst, inst->getOperand(1));
        auto lhs = useValue(inst, inst->getOperand(0));
        return std::string("(store.") + std::string(typeSuffix(function->global->module, inst->getOperand(0)->getType()))
            + " " + lhs + " " + rhs + ")";
    }
    case Instruction::Br: {
        auto result = std::string("");
        auto &B = cast<BranchInst>(*inst);
        // TODO: do cmp fusing here - peer through or.i1/and.i1 instructions too.
        if(B.isConditional()) {
            // TODO: Examine or.i1 and and.i1 instructions too.
            // These can be transformed, but care must be taken to avoid short-circuiting.
            if(isa<CmpInst>(B.getCondition()) && maybeFuseValue(inst, B.getCondition())) {
                auto &CI = cast<CmpInst>(*B.getCondition());
                // Proper evaluation order...
                auto rhs = useValue(&CI, CI.getOperand(1));
                auto lhs = useValue(&CI, CI.getOperand(0));
                result += std::string("(if (") + std::string(CI.getOpcodeName(CI.getOpcode())) + "." + getPredicateText(CI.getPredicate()) + ".fused." + std::string(typeSuffix(function->global->module, CI.getOperand(0)->getType()))
                    + " " + lhs + " " + rhs + ")\n";
            } else {
                result += "(if (not (eql " + useValue(inst, B.getCondition()) + " 0))\n";
            }
            result += "            (progn\n";
            result += prepareForBranch(inst, B.getSuccessor(0), "              ", false);
            result += "              (go " + std::string(function->name_table[B.getSuccessor(0)]) + "))\n";
            result += "            (progn\n";
            result += prepareForBranch(inst, B.getSuccessor(1), "              ", false);
            result += "              (go " + std::string(function->name_table[B.getSuccessor(1)]) + ")))";
        } else {
            result += prepareForBranch(inst, B.getSuccessor(0), "", true);
            if(isa<PHINode>(B.getSuccessor(0)->front())) {
                result += "        ";
            }
            result += "(go " + std::string(function->name_table[B.getSuccessor(0)]) + ")";
        }
        return result;
    }
    case Instruction::Ret: {
        auto &ret = cast<ReturnInst>(*inst);
        auto val = ret.getReturnValue();
        if(val) {
            return "(return " + useValue(inst, val) + ")";
        } else {
            return "(return)";
        }
    }
    case Instruction::Unreachable: {
        return "(error \"Reached unreachable!\")";
    }
    case Instruction::Call: {
        auto &call = cast<CallInst>(*inst);
        if(call.isInlineAsm()) {
            errs() << "Unsupported instruction " << inst << "\n";
            report_fatal_error("Inline asm? You must be joking!");
        }
        if(isSetjmpCall(*inst)) {
            return "";
        }
        auto result = std::string("");
        // Work in reverse to ensure operands are evaluated in the right order.
        std::vector<std::string> operands(call.getNumArgOperands());
        for(unsigned i = call.getNumArgOperands(); i != 0; i--) {
            operands[i-1] = useValue(inst, call.getArgOperand(i-1));
        }
        if(auto target = call.getCalledFunction()) {
            result += "(call-direct " + escapeSymbol(target->getName());
        } else {
            result += "(call-indirect " + useValue(inst, call.getCalledValue());
        }
        for(auto &operand: operands) {
            result += " " + operand;
        }
        result += ")";
        return result;
    }
    case Instruction::Select: {
        auto &select = cast<SelectInst>(*inst);
        // TODO: Try fusing conditionals here.
        // evaluation order.
        auto false_val = useValue(inst, select.getFalseValue());
        auto true_val = useValue(inst, select.getTrueValue());
        auto cond_val = useValue(inst, select.getCondition());
        return std::string("(select ") + cond_val + " " + true_val + " " + false_val + ")";
    }
    case Instruction::GetElementPtr: {
        auto &DL = function->global->module->getDataLayout();
        auto &GEP = cast<GetElementPtrInst>(*inst);
        if(!GEP.hasAllConstantIndices()) {
            errs() << "Unsupported instruction " << (*inst) << "\n";
            report_fatal_error(std::string("Unsupported instruction ") +
                               inst->getOpcodeName(inst->getOpcode()) + " (with non-constant indicies)");
        }
        APInt offset(DL.getPointerSizeInBits(GEP.getPointerAddressSpace()), 0);
        if(!GEP.accumulateConstantOffset(DL, offset)) {
            report_fatal_error("Unable to compute GEP offset.");
        }
        if(offset == 0) {
            return useValue(inst, GEP.getPointerOperand());
        } else {
            return std::string("(add.") + std::string(typeSuffix(function->global->module, GEP.getType()))
                + " " + useValue(inst, GEP.getPointerOperand()) + " " + offset.toString(10, false) + ")";
        }
    }
    default:
        errs() << "Unsupported instruction " << *inst << "\n";
        report_fatal_error(std::string("Unsupported instruction ") +
                           inst->getOpcodeName(inst->getOpcode()));
    }
}

void IotaFunctionTranslator::translateBlock(BasicBlock &bb, std::unique_ptr<tool_output_file> &Out) {
    IotaBlockTranslator block_xlat;
    block_xlat.function = this;
    for(auto &inst: bb) {
        if(!isa<PHINode>(inst)) {
            block_xlat.remaining.push_back(&inst);
        }
    }
    std::vector<std::pair<Instruction *, std::string>> entries;
    // Working backwards.
    while(!block_xlat.remaining.empty()) {
        auto inst = block_xlat.remaining.back();
        block_xlat.remaining.pop_back();
        entries.push_back(std::make_pair(inst, block_xlat.translateInstruction(inst)));
    }
    // Work backwards again to emit them in the right order.
    std::string close_parens = "";
    std::vector<std::string> pending_type_decls;
    bool in_let = false;
    while(!entries.empty()) {
        auto entry = entries.back();
        entries.pop_back();
        auto inst = entry.first;
        auto code = entry.second;
        if(isSetjmpCall(*inst)) {
            if(in_let) {
                // Close up the current LET*
                close_parens += ")";
                Out->os() << ")\n         " << "(declare ";
                for(auto &decl: pending_type_decls) {
                    Out->os() << decl << "\n";
                }
                Out->os() << ")\n";
                pending_type_decls.clear();
                in_let = false;
            }
            auto &call = cast<CallInst>(*inst);
            Out->os() << "(tagbody\n";
            // Generate a setjmp thunk variable for this instruction.
            Out->os() << "        (setf setjmp-thunk." << name_table[inst] << " (setjmp.prepare setjmp-target." << name_table[inst] << "))\n";
            Out->os() << "(call-direct " << escapeSymbol(call.getCalledFunction()->getName()) << " setjmp-thunk." + name_table[inst];
            for(auto &operand: call.arg_operands()) {
                Out->os() << " " << valueForPrinting(operand);
            }
            Out->os() << ")\n";
            Out->os() << "        (setq " << name_table[inst] << " 0)\n";
            Out->os() << "        (go setjmp-resume." << name_table[inst] << ")\n";
            Out->os() << "      setjmp-target." << name_table[inst] << "\n";
            Out->os() << "        (setq " << name_table[inst] << " 1)\n";
            Out->os() << "        (go setjmp-resume." << name_table[inst] << ")\n";
            Out->os() << "      setjmp-resume." << name_table[inst] << "\n";
            close_parens += ")"; // close up the tagbody.
        } else if(inst->getType()->isVoidTy() ||
                  inst->use_empty()) {
            if(in_let) {
                // Close up the current LET*
                close_parens += ")";
                Out->os() << ")\n         " << "(declare ";
                for(auto &decl: pending_type_decls) {
                    Out->os() << decl << "\n";
                }
                Out->os() << ")\n";
                pending_type_decls.clear();
                in_let = false;
            }
            if(entries.empty() && !dom[&bb]->getChildren().empty()) {
                // This is the block's terminator instruction.
                // Any blocks that this block immediately dominates will be emitted after
                // it, but in a tagbody with the right scope. Emit the tagbody here
                // so labels are visible.
                Out->os() << "(tagbody\n";
                close_parens += ")";
            }
            // nothing - result not used.
            Out->os() << "         " << code << "\n";
        } else {
            // Local to the basic block and its children in the dominator tree.
            if(in_let) {
                Out->os() << "\n         " << "       (" << name_table[inst] << " " << code << ")";
            } else {
                Out->os() << "         " << "(let* ((" << name_table[inst] << " " << code << ")";
                in_let = true;
            }
            if(EmitTypeDeclarartions) {
                pending_type_decls.push_back(std::string("(type ") + std::string(clTypeName(global->module, inst->getType())) + " " + name_table[inst] + ")");
            }
        }
    }
    // Now emit any child blocks.
    // TODO: Be more clever about this. BR could be changed to emit the block inline if it one has one predecessor
    // and the BR dominates the target.
    for(auto &child: *dom[&bb]) {
        Out->os() << "       " << name_table[child->getBlock()] << "\n";
        translateBlock(*child->getBlock(), Out);
    }
    if(!close_parens.empty()) {
        Out->os() << close_parens << "\n";
    }
}

static void translateFunction(IotaTranslator *xlat, Function &fn, std::unique_ptr<tool_output_file> &Out) {
    IotaFunctionTranslator function_xlat;
    function_xlat.global = xlat;
    function_xlat.dom = DominatorTreeAnalysis().run(fn);

    Out->os() << "(define-llvm-function " << escapeSymbol(fn.getName()) << " ((";
    std::unordered_map<Value*, std::string> &name_table = function_xlat.name_table;
    for(auto &arg: fn.args()) {
        std::string name = arg.getName();
        if(name.empty()) {
            name += "%";
        } else {
            name += ".";
        }
        name += std::to_string(name_table.size());
        name_table[&arg] = escapeSymbol(name);
        Out->os() << escapeSymbol(name) << " ";
    }
    Out->os() << ")";
    bool need_frame_pointer = false;
    for(auto &inst: fn.getEntryBlock()) {
        if(isa<AllocaInst>(inst)) {
            need_frame_pointer = true;
            break;
        }
    }
    Out->os() << " :need-frame-pointer " << (need_frame_pointer ? "t" : "nil");
    if(fn.callsFunctionThatReturnsTwice()) {
        Out->os() << " :uses-setjmp t";
    }
    Out->os() << ")\n";
    if(!OptimizeQualities.empty()) {
        Out->os() << "  (declare (optimize " << OptimizeQualities << "))\n";
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
    // Create names for each instruction.
    for(auto &bb: fn) {
        for(auto &inst: bb) {
            auto ty = inst.getType();
            if(ty->isVoidTy()) {
                continue;
            }
            if(inst.use_empty()) {
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
        }
    }
    // Create variables for PHINodes.
    Out->os() << "  (let (";
    for(auto &bb: fn) {
        for(auto &inst: bb) {
            auto ty = inst.getType();
            if(ty->isVoidTy()) {
                continue;
            }
            std::string &name = name_table[&inst];
            if(isSetjmpCall(inst)) {
                // Generate a setjmp thunk variable for this instruction.
                // And also the variable to hold the setjmp result
                Out->os() << "(setjmp-thunk." << name << " nil)\n        ";
                Out->os() << "(" << name << " nil)\n        ";
            }
            if(!isa<PHINode>(inst)) {
                continue;
            }
            Out->os() << "(" << name << " " << clTypeInitializer(ty) << ")\n        ";
        }
    }
    Out->os() << ")\n";
    if(EmitTypeDeclarartions) {
        Out->os() << "    (declare ";
        for(auto &arg: fn.args()) {
            auto ty = arg.getType();
            Out->os() << "(type " << clTypeName(xlat->module, ty) << " " << name_table[&arg] << ")\n             ";
        }
        for(auto &bb: fn) {
            for(auto &inst: bb) {
                auto ty = inst.getType();
                if(ty->isVoidTy()) {
                    continue;
                }
                if(!isa<PHINode>(inst)) {
                    continue;
                }
                Out->os() << "(type " << clTypeName(xlat->module, ty) << " " << name_table[&inst] << ")\n             ";
            }
        }
        Out->os() << ")\n";
    }
    Out->os() << "    (block nil\n";
    Out->os() << "      (tagbody\n";
    // Generate code for each basic block.
    // Traverse the dominator tree to get nesting right.
    // TODO: This does not work right for particularly perverse uses of setjmp,
    // where the call to setjmp does not dominate all calls to longjmp.
    // These kinds of uses will manifest as GOs to expired GO tags.
    // FIXME: Can instructions jump to the entry block? Don't think so...
    function_xlat.translateBlock(fn.getEntryBlock(), Out);
    Out->os() << "))))\n\n";
}

static void storeInteger(std::vector<uint8_t> &data_section, uint64_t offset, uint64_t value, int width) {
    switch(width) {
    case 64:
        data_section[offset+7] = value >> 56;
        data_section[offset+6] = value >> 48;
        data_section[offset+5] = value >> 40;
        data_section[offset+4] = value >> 32;
        [[gnu::fallthrough]];
    case 32:
        data_section[offset+3] = value >> 24;
        data_section[offset+2] = value >> 16;
        [[gnu::fallthrough]];
    case 16:
        data_section[offset+1] = value >> 8;
        [[gnu::fallthrough]];
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
        uint64_t numeric_value = resolveConstantInteger(&*module, global_table, value);
        storeInteger(data_section, offset, numeric_value,
                     module->getDataLayout().getTypeStoreSizeInBits(value->getType()));
    }
}

static void iotaTranslate(std::unique_ptr<Module> &module, std::unique_ptr<tool_output_file> &Out) {
    IotaTranslator xlat;
    xlat.module = &*module;

    unsigned long long data_origin = 0x200000;
    unsigned long long data_end = data_origin;
    unsigned long long next_fn_id = 1;
    // Lay out the data section.
    std::unordered_map<GlobalObject *, unsigned long long> &global_table = xlat.global_table;
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
        Out->os() << "(in-package " << PackageName << ")\n\n";
    }

    Out->os() << "(defun make-context (&rest personality-initargs)\n";
    Out->os() << "  (make-llvm-context\n";
    if(ContextPersonality.empty()) {
        Out->os() << "   :unix\n";
    } else {
        Out->os() << "   '" << ContextPersonality << "\n";
    }
    Out->os() << "   " << data_origin << "\n";
    Out->os() << "   #.(make-array " << (data_end - data_origin) << " :element-type '(unsigned-byte 8) :initial-contents '(";
    for(size_t i = 0; i < data_section.size(); i += 1) {
        if((i % 64) == 0) {
            Out->os() << "\n     ";
        } else {
            Out->os() << " ";
        }
        Out->os() << (int)data_section[i];
    }
    Out->os() << "))\n";
    Out->os() << "   " << (module->getDataLayout().getPointerSizeInBits(0) == 32 ? "t" : "nil") << "\n";
    Out->os() << "   #(";
    for(size_t i = 0; i < function_table.size(); i += 1) {
        if((i % 64) == 0) {
            Out->os() << "\n     ";
        } else {
            Out->os() << " ";
        }
        Out->os() << escapeSymbol(function_table[i]->getName());
    }
    Out->os() << ")\n";
    if(LibraryMode && EntryPoint == "_start") {
        Out->os() << "   nil\n";
    } else {
        Out->os() << "   '" << escapeSymbol(EntryPoint) << "\n";
    }
    Out->os() << "   personality-initargs))\n\n";

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
            translateFunction(&xlat, fn, Out);
        }
    }
}

static void diagnosticHandler(const DiagnosticInfo &DI) {
  unsigned Severity = DI.getSeverity();
  switch (Severity) {
  case DS_Error:
    errs() << "ERROR: ";
    break;
  case DS_Warning:
    errs() << "WARNING: ";
    break;
  case DS_Remark:
  case DS_Note:
    llvm_unreachable("Only expecting warnings and errors");
  }

  DiagnosticPrinterRawOStream DP(errs());
  DI.print(DP);
  errs() << '\n';
}

ErrorOr<object::OwningBinary<object::Binary>> createBinary(StringRef Path, LLVMContext *Context) {
    using namespace object;
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(Path);
  if (std::error_code EC = FileOrErr.getError())
    return EC;
  std::unique_ptr<MemoryBuffer> &Buffer = FileOrErr.get();

  ErrorOr<std::unique_ptr<Binary>> BinOrErr =
      createBinary(Buffer->getMemBufferRef(), Context);
  if (std::error_code EC = BinOrErr.getError())
    return EC;
  std::unique_ptr<Binary> &Bin = BinOrErr.get();

  return OwningBinary<Binary>(std::move(Bin), std::move(Buffer));
}

static bool readInputFile(StringRef Path, LLVMContext *Context, Linker *L) {
    auto BinaryOrErr = createBinary(Path, Context);
    if (std::error_code ec = BinaryOrErr.getError()) {
        errs() << Path << ": " << ec.message() << '\n';
        return false;
    }
    object::Binary &Bin = *BinaryOrErr.get().getBinary();

    if (object::IRObjectFile *IR = dyn_cast<object::IRObjectFile>(&Bin)) {
        auto &M = IR->getModule();
        if (verifyModule(M, &errs())) {
            errs() << Path << ": error: input module is broken!\n";
            return false;
        }

        if (L->linkInModule(&M))
            return false;
    } else if (object::Archive *Archive = dyn_cast<object::Archive>(&Bin)) {
        for(auto &ArchiveChild: Archive->children()) {
            auto ChildOrErr = ArchiveChild.getAsBinary(Context);
            if (std::error_code ec = ChildOrErr.getError()) {
                errs() << Path << ": " << ArchiveChild.getRawName() << ": "
                       << ec.message() << '\n';
                return false;
            }
            if (object::IRObjectFile *ChildIR = dyn_cast<object::IRObjectFile>(&*ChildOrErr.get())) {
                auto &M = ChildIR->getModule();
                if (verifyModule(M, &errs())) {
                    errs() << Path << ": " << ArchiveChild.getRawName()
                           << ": error: input module is broken!\n";
                    return false;
                }

                if (L->linkInModule(&M))
                    return false;
            } else {
                errs() << Path << ": "
                       << ArchiveChild.getRawName()
                       << ": unrecognizable file type\n";
                return false;
            }
        }
    } else {
        errs() << Path << ": unrecognizable file type\n";
        return false;
    }
    return true;
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

    if (OutputFilename.empty()) {
        OutputFilename = "-";
    }

    std::error_code OutEC;
    std::unique_ptr<tool_output_file> Out(
        new tool_output_file(OutputFilename, OutEC, sys::fs::F_None));
    if (OutEC) {
      errs() << OutEC.message() << '\n';
      return -1;
    }

    LLVMContext context;

    auto module = make_unique<Module>("iota", context);
    Linker L(module.get(), diagnosticHandler);

    // Read in ordinary input files.
    for (auto &Path: InputFilenames) {
        if(!readInputFile(Path, &context, &L)) {
            return -1;
        }
    }

    // Read in libraries.
    for (auto &Name: InputLibraries) {
        std::string fullName = "lib" + Name + ".a";
        bool found = false;
        for (auto &Dir: LibrarySearchPaths) {
            SmallString<128> P(Dir);
            llvm::sys::path::append(P, fullName);
            if (llvm::sys::fs::exists(Twine(P))) {
                found = true;
                if(!readInputFile(P, &context, &L)) {
                    return -1;
                }
                break;
            }
        }
        if (!found) {
            errs() << "Can't find library " << Name << '\n';
            return -1;
        }
    }

    if(!TranslateOnly) {
        lowerVectorBswap(*module);

        legacy::PassManager PM;

        // Internalize all symbols in the module except the entry point.
        if(LibraryMode) {
            PM.add(createInternalizePass());
        } else {
            const char *export_name = EntryPoint.c_str();
            PM.add(createInternalizePass(export_name));
        }

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

    iotaTranslate(module, Out);
    Out->keep();
    Out->os().close();

    return 0;
}
