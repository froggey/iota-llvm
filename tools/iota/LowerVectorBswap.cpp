#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

// This "pass" lowers calls to the bswap intrinsic with a vector argument
// to a number of calls to scalar bswap.

// FIXME: Turn this into a proper pass.

static bool lowerVectorBSwap(Module *M, Instruction *I) {
    auto CI = dyn_cast<CallInst>(I);
    if(!CI) {
        return false;
    }
    auto Callee = CI->getCalledFunction();
    if(!Callee) {
        return false;
    }
    if(Callee->getIntrinsicID() != Intrinsic::bswap) {
        return false;
    }
    if(!isa<VectorType>(CI->getType())) {
        return false;
    }
    IRBuilder<> Builder(CI);

    auto V = CI->getOperand(0);
    auto Ty = cast<VectorType>(CI->getType());
    auto ordinaryBSwap = Intrinsic::getDeclaration(M,
                                                   Intrinsic::bswap,
                                                   Ty->getElementType());
    Value *result = UndefValue::get(Ty);
    for(unsigned i = 0; i < Ty->getNumElements(); i += 1) {
        auto index = ConstantInt::get(IntegerType::get(M->getContext(), 32),
                                      i);
        auto elt = Builder.CreateExtractElement(V, index);
        auto swapped = Builder.CreateCall(ordinaryBSwap, elt);
        result = Builder.CreateInsertElement(result, swapped, index);
    }
    CI->replaceAllUsesWith(result);
    return true;
}

static void hackFunction(Module *M, Function &F) {
    if(F.empty()) {
        return;
    }
    SmallVector<Instruction *, 8> remove_me;
    for(auto &BB: F) {
        for(auto &I: BB) {
            if(lowerVectorBSwap(M, &I)) {
                remove_me.push_back(&I);
            }
        }
    }
    for(auto I: remove_me) {
        I->eraseFromParent();
    }
}

void lowerVectorBswap(Module &M) {
  for(auto &fn : M) {
      hackFunction(&M, fn);
  }
}
