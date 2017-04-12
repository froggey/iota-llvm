#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

// This "pass" lowers
//  %1 = bitcast <4 x i32> %0 to i128
//  %2 = trunc i128 %1 to i32
// to an extractelement instruction.
// Scalarize doesn't catch this.

// FIXME: Turn this into a proper pass.

static bool lowerBitcastTrunc(Module *M, Instruction *I) {
    auto TI = dyn_cast<TruncInst>(I);
    if(!TI) {
        return false;
    }
    if(TI->getType() != IntegerType::get(M->getContext(), 32)) {
        return false;
    }
    if(TI->getOperand(0)->getType() != IntegerType::get(M->getContext(), 128)) {
        return false;
    }
    auto BCI = dyn_cast<BitCastInst>(TI->getOperand(0));
    if(!BCI) {
        return false;
    }
    auto VecTy = dyn_cast<VectorType>(BCI->getOperand(0)->getType());
    if(!VecTy) {
        return false;
    }
    // This is a sequence of:
    //  %1 = bitcast <4 x i32> %0 to i128
    //  %2 = trunc i128 %1 to i32
    // Replace with extractelement 0.
    IRBuilder<> Builder(TI);
    auto index = ConstantInt::get(IntegerType::get(M->getContext(), 32),
                                  0);
    auto result = Builder.CreateExtractElement(BCI->getOperand(0), index);
    TI->replaceAllUsesWith(result);
    return true;
}

static void hackFunction(Module *M, Function &F) {
    if(F.empty()) {
        return;
    }
    SmallVector<Instruction *, 8> remove_me;
    for(auto &BB: F) {
        for(auto &I: BB) {
            if(lowerBitcastTrunc(M, &I)) {
                remove_me.push_back(&I);
            }
        }
    }
    for(auto I: remove_me) {
        I->eraseFromParent();
    }
}

void lowerBitcastTrunc(Module &M) {
  for(auto &fn : M) {
      hackFunction(&M, fn);
  }
}
