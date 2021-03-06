//===-- llvm-dis.cpp - The low-level LLVM disassembler --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility may be invoked in the following manner:
//  llvm-dis [options]      - Read LLVM bitcode from stdin, write asm to stdout
//  llvm-dis [options] x.bc - Read LLVM bitcode from the x.bc file, write asm
//                            to the x.ll file.
//  Options:
//      --help   - Output information about command line switches
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LLVMContext.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Bitcode/NaCl/NaClReaderWriter.h"  // @LOCALMOD
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IRReader/IRReader.h"  // @LOCALMOD
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataStream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StreamingMemoryObject.h" // @LOCALMOD
#include "llvm/Support/ToolOutputFile.h"
#include <system_error>
using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bitcode>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"),
               cl::value_desc("filename"));

static cl::opt<bool>
Force("f", cl::desc("Enable binary output on terminals"));

static cl::opt<bool>
DontPrint("disable-output", cl::desc("Don't output the .ll file"), cl::Hidden);

static cl::opt<bool>
ShowAnnotations("show-annotations",
                cl::desc("Add informational comments to the .ll file"));

static cl::opt<bool> PreserveAssemblyUseListOrder(
    "preserve-ll-uselistorder",
    cl::desc("Preserve use-list order when writing LLVM assembly."),
    cl::init(false), cl::Hidden);

// @LOCALMOD-BEGIN
static cl::opt<NaClFileFormat>
InputFileFormat(
    "bitcode-format",
    cl::desc("Define format of input bitcode file:"),
    cl::values(
        clEnumValN(LLVMFormat, "llvm", "LLVM bitcode file (default)"),
        clEnumValN(PNaClFormat, "pnacl", "PNaCl bitcode file"),
        clEnumValEnd),
    cl::init(LLVMFormat));
// @LOCALMOD-END

namespace {

static void printDebugLoc(const DebugLoc &DL, formatted_raw_ostream &OS) {
  OS << DL.getLine() << ":" << DL.getCol();
  if (MDLocation *IDL = DL.getInlinedAt()) {
    OS << "@";
    printDebugLoc(IDL, OS);
  }
}
class CommentWriter : public AssemblyAnnotationWriter {
public:
  void emitFunctionAnnot(const Function *F,
                         formatted_raw_ostream &OS) override {
    OS << "; [#uses=" << F->getNumUses() << ']';  // Output # uses
    OS << '\n';
  }
  void printInfoComment(const Value &V, formatted_raw_ostream &OS) override {
    bool Padded = false;
    if (!V.getType()->isVoidTy()) {
      OS.PadToColumn(50);
      Padded = true;
      OS << "; [#uses=" << V.getNumUses() << " type=" << *V.getType() << "]";  // Output # uses and type
    }
    if (const Instruction *I = dyn_cast<Instruction>(&V)) {
      if (const DebugLoc &DL = I->getDebugLoc()) {
        if (!Padded) {
          OS.PadToColumn(50);
          Padded = true;
          OS << ";";
        }
        OS << " [debug line = ";
        printDebugLoc(DL,OS);
        OS << "]";
      }
      if (const DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(I)) {
        DIVariable Var(DDI->getVariable());
        if (!Padded) {
          OS.PadToColumn(50);
          OS << ";";
        }
        OS << " [debug variable = " << Var->getName() << "]";
      }
      else if (const DbgValueInst *DVI = dyn_cast<DbgValueInst>(I)) {
        DIVariable Var(DVI->getVariable());
        if (!Padded) {
          OS.PadToColumn(50);
          OS << ";";
        }
        OS << " [debug variable = " << Var->getName() << "]";
      }
    }
  }
};

} // end anon namespace

static void diagnosticHandler(const DiagnosticInfo &DI, void *Context) {
  raw_ostream &OS = errs();
  OS << (char *)Context << ": ";
  switch (DI.getSeverity()) {
  case DS_Error: OS << "error: "; break;
  case DS_Warning: OS << "warning: "; break;
  case DS_Remark: OS << "remark: "; break;
  case DS_Note: OS << "note: "; break;
  }

  DiagnosticPrinterRawOStream DP(OS);
  DI.print(DP);
  OS << '\n';

  if (DI.getSeverity() == DS_Error)
    exit(1);
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  LLVMContext &Context = getGlobalContext();
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  Context.setDiagnosticHandler(diagnosticHandler, argv[0]);

  cl::ParseCommandLineOptions(argc, argv, "llvm .bc -> .ll disassembler\n");

  bool FoundErrors = false; // @LOCALMOD
  std::string ErrorMessage;
  std::unique_ptr<Module> M;

  // Use the bitcode streaming interface
  DataStreamer *Streamer = getDataFileStreamer(InputFilename, &ErrorMessage);
  if (Streamer) {
    std::unique_ptr<StreamingMemoryObject> Buffer(
        new StreamingMemoryObjectImpl(Streamer));  // @LOCALMOD
    std::string DisplayFilename;
    if (InputFilename == "-")
      DisplayFilename = "<stdin>";
    else
      DisplayFilename = InputFilename;

    // @LOCALMOD-BEGIN
    switch (InputFileFormat) {
      case LLVMFormat: {
        // The Module's BitcodeReader's BitstreamReader takes ownership
        // of the StreamingMemoryObject.
        ErrorOr<std::unique_ptr<Module>> MOrErr =
            getStreamedBitcodeModule(DisplayFilename, Buffer.release(),
                                     Context);
        M = std::move(*MOrErr);
        M->materializeAllPermanently();
        break;
      }
      case PNaClFormat: {
        M.reset(getNaClStreamedBitcodeModule(
            DisplayFilename, Buffer.release(), Context,
            /* DiagnosticHandler = */ nullptr, &ErrorMessage));
        if(M.get()) {
          if (std::error_code EC = M->materializeAllPermanently()) {
            FoundErrors = true;
            ErrorMessage = EC.message();
            M.reset();
          }
        } else {
          FoundErrors = true;
        }
        break;
      }
      case AutodetectFileFormat:
        report_fatal_error("Command can't autodetect file format!");
    }
  } else {
    FoundErrors = true;
  }
  if (FoundErrors) {
    if (ErrorMessage.empty())
      ErrorMessage = "bitcode didn't read correctly.";
    errs() << argv[0] << ": " << ErrorMessage << '\n';
    return 1;
  }
  // @LOCALMOD-END

  // Just use stdout.  We won't actually print anything on it.
  if (DontPrint)
    OutputFilename = "-";

  if (OutputFilename.empty()) { // Unspecified output, infer it.
    if (InputFilename == "-") {
      OutputFilename = "-";
    } else {
      const std::string &IFN = InputFilename;
      int Len = IFN.length();
      // If the source ends in .bc, strip it off.
      if (IFN[Len-3] == '.' && IFN[Len-2] == 'b' && IFN[Len-1] == 'c')
        OutputFilename = std::string(IFN.begin(), IFN.end()-3)+".ll";
      else
        OutputFilename = IFN+".ll";
    }
  }

  std::error_code EC;
  std::unique_ptr<tool_output_file> Out(
      new tool_output_file(OutputFilename, EC, sys::fs::F_None));
  if (EC) {
    errs() << EC.message() << '\n';
    return 1;
  }

  std::unique_ptr<AssemblyAnnotationWriter> Annotator;
  if (ShowAnnotations)
    Annotator.reset(new CommentWriter());

  // All that llvm-dis does is write the assembly to a file.
  if (!DontPrint)
    M->print(Out->os(), Annotator.get(), PreserveAssemblyUseListOrder);

  // Declare success.
  Out->keep();

  return 0;
}
