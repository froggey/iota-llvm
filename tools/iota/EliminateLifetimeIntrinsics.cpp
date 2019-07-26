#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

// This "pass" elimiates calls to the llvm.lifetime.start/.end intrinsics.
// They are not needed for code generation and complicate the resulting output.

// FIXME: Turn this into a proper pass.

static bool eliminateLifetimeIntrinsics(Module *M, Instruction *I) {
    auto CI = dyn_cast<CallInst>(I);
    if(!CI) {
        return false;
    }
    auto Callee = CI->getCalledFunction();
    if(!Callee) {
        return false;
    }
    if(Callee->getIntrinsicID() != Intrinsic::lifetime_end &&
       Callee->getIntrinsicID() != Intrinsic::lifetime_start) {
        return false;
    }
    return true;
}

static void hackFunction(Module *M, Function &F) {
    if(F.empty()) {
        return;
    }
    SmallVector<Instruction *, 8> remove_me;
    for(auto &BB: F) {
        for(auto &I: BB) {
            if(eliminateLifetimeIntrinsics(M, &I)) {
                remove_me.push_back(&I);
            }
        }
    }
    for(auto I: remove_me) {
        I->eraseFromParent();
    }
}

void eliminateLifetimeIntrinsics(Module &M) {
  for(auto &fn : M) {
      hackFunction(&M, fn);
  }
}
