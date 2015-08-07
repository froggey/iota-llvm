//===- X86MCNaClExpander.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the X86MCNaClExpander class, the X86 specific
// subclass of MCNaClExpander.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_MC_X86MCNACLEXPANDER_H
#define LLVM_MC_X86MCNACLEXPANDER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCNaClExpander.h"
#include "llvm/MC/MCRegisterInfo.h"

namespace llvm {
class MCContext;
class MCStreamer;
class MCSubtargetInfo;

namespace X86 {
class X86MCNaClExpander : public MCNaClExpander {
public:
  X86MCNaClExpander(const MCContext &Ctx, std::unique_ptr<MCRegisterInfo> &&RI,
                    std::unique_ptr<MCInstrInfo> &&II, bool Is64Bit)
      : MCNaClExpander(Ctx, std::move(RI), std::move(II)), Is64Bit(Is64Bit) {}

  bool expandInst(const MCInst &Inst, MCStreamer &Out,
                  const MCSubtargetInfo &STI) override;
protected:
  bool isValidScratchRegister(unsigned Reg) const override;

private:
  bool Guard = false; // recursion guard
  bool Is64Bit = false;
  SmallVector<MCInst, 4> Prefixes;

  void emitPrefixes(MCStreamer &Out, const MCSubtargetInfo &STI);

  void expandIndirectBranch(const MCInst &Inst, MCStreamer &Out,
                            const MCSubtargetInfo &STI);

  void expandReturn(const MCInst &Inst, MCStreamer &Out,
                    const MCSubtargetInfo &STI);

  void expandLoadStore(const MCInst &Inst, MCStreamer &Out,
                       const MCSubtargetInfo &STI, bool EmitPrefixes);

  void expandStringOperation(const MCInst &Inst, MCStreamer &Out,
			     const MCSubtargetInfo &STI, bool EmitPrefixes);

  void doExpandInst(const MCInst &Inst, MCStreamer &Out,
                    const MCSubtargetInfo &STI, bool EmitPrefixes);

  void expandExplicitStackManipulation(unsigned StackReg, const MCInst &Inst,
                                       MCStreamer &Out,
                                       const MCSubtargetInfo &STI,
                                       bool EmitPrefixes);

  void emitSandboxMemOp(MCInst &Inst, int MemIdx, unsigned ScratchReg,
                        MCStreamer &Out, const MCSubtargetInfo &STI);

  bool emitSandboxMemOps(MCInst &Inst, unsigned ScratchReg, MCStreamer &Out,
                         const MCSubtargetInfo &STI);

  void emitInstruction(const MCInst &Inst, MCStreamer &Out,
                       const MCSubtargetInfo &STI, bool EmitPrefixes);

  void emitSandboxBranchReg(unsigned Reg, MCStreamer &Out,
                            const MCSubtargetInfo &STI);
  void emitIndirectJumpReg(unsigned Reg, MCStreamer &Out,
                           const MCSubtargetInfo &STI);
  void emitIndirectCallReg(unsigned Reg, MCStreamer &Out,
                           const MCSubtargetInfo &STI);
};
}
}
#endif
