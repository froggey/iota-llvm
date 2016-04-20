//===- ConvertToPSO.cpp - Convert module to a PNaCl PSO--------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The ConvertToPSO pass is part of an implementation of dynamic
// linking for PNaCl.  It transforms an LLVM module to be a PNaCl PSO
// (portable shared object).
//
// This pass takes symbol information that's stored at the LLVM IR
// level and moves it to be stored inside variables within the module,
// in a data structure rooted at the "__pnacl_pso_root" variable.
//
// This means that when the module is dynamically loaded, a runtime
// dynamic linker can read the "__pnacl_pso_root" data structure to
// look up symbols that the module exports and supply definitions of
// symbols that a module imports.
//
// Currently, this pass implements:
//
//  * Exporting symbols
//  * Importing symbols
//     * when referenced by global variable initializers
//     * when referenced by functions
//  * Building a hash table of exported symbols to allow O(1)-time lookup
//  * Support for thread-local variables (module-local use only)
//  * Support for exporting and importing thread-local variables
//
// The following features are not implemented yet:
//
//  * Support for lazy binding (i.e. lazy symbol resolution)
//
//===----------------------------------------------------------------------===//

#include "ExpandTls.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/NaCl.h"

using namespace llvm;

namespace {
  // This version number can be incremented when the format of the PSO data
  // is changed in an incompatible way.
  //
  // For the time being, this is intended only as a convenience for making
  // cross-repo changes, because the PSO format is interpreted by code in
  // the native_client repo.  The PSO format is not intended to be stable
  // yet.
  //
  // If the format is changed in a compatible way, an alternative is to
  // increment TOOLCHAIN_FEATURE_VERSION instead.
  const int PSOFormatVersion = 2;

  // Command line arguments consist of a list of PLL dependencies.
  static cl::list<std::string>
  LibraryDependencies("convert-to-pso-deps",
                      cl::CommaSeparated,
                      cl::desc("The dependencies of the PLL being built"),
                      cl::value_desc("PLL to list as dependency"));

  // This is a ModulePass because it inherently operates on a whole module.
  class ConvertToPSO : public ModulePass {
  public:
    static char ID; // Pass identification, replacement for typeid
    ConvertToPSO() : ModulePass(ID) {
      initializeConvertToPSOPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnModule(Module &M);
  };

  // This is Dan Bernstein's string hash algorithm.
  uint32_t hashString(const std::string &S) {
    uint32_t H = 5381;
    for (unsigned char Ch : S)
      H = H * 33 + Ch;
    return H;
  }

  class SymbolTableEntry {
  public:
    SymbolTableEntry(StringRef Name, Constant *Val) :
        Name(Name.data()), Value(Val), Hash(hashString(Name)) {}

    std::string Name;
    Constant *Value;
    uint32_t Hash;
  };

  // This takes a SimpleElement from FlattenGlobals' normal form.  If the
  // SimpleElement is a reference to a GlobalValue, it returns the
  // GlobalValue along with its addend.  Otherwise, it returns nullptr.
  GlobalValue *getReference(Constant *Init, uint64_t *Addend) {
    *Addend = 0;
    if (isa<ArrayType>(Init->getType()))
      return nullptr;
    if (auto CE = dyn_cast<ConstantExpr>(Init)) {
      if (CE->getOpcode() == Instruction::Add) {
        if (auto CI = dyn_cast<ConstantInt>(CE->getOperand(1))) {
          if (auto Op0 = dyn_cast<ConstantExpr>(CE->getOperand(0))) {
            CE = Op0;
            *Addend = CI->getSExtValue();
          }
        }
      }
      if (CE->getOpcode() == Instruction::PtrToInt) {
        if (auto GV = dyn_cast<GlobalValue>(CE->getOperand(0))) {
          if (!GV->isDeclaration())
            return nullptr;
          return GV;
        }
      }
    }
    errs() << "Initializer value not handled: " << *Init << "\n";
    report_fatal_error("ConvertToPSO: Value is not a SimpleElement");
  }

  // Set up an array as a Global Variable, given a SmallVector.
  Constant *createArray(Module &M, const char *Name,
                        SmallVectorImpl<Constant *> *Array,
                        Type *ElementType) {
    Constant *Contents = ConstantArray::get(
        ArrayType::get(ElementType, Array->size()), *Array);
    return new GlobalVariable(
        M, Contents->getType(), true, GlobalValue::InternalLinkage,
        Contents, Name);
  }

  Constant *createDataArray(Module &M, const char *Name,
                            SmallVectorImpl<uint32_t> *Array) {
    Constant *Contents = ConstantDataArray::get(M.getContext(), *Array);
    return new GlobalVariable(
        M, Contents->getType(), true, GlobalValue::InternalLinkage,
        Contents, Name);
  }

  // This function adds a level of indirection to references by functions
  // to imported GlobalValues.  Any time a function refers to a symbol that
  // is defined outside the module, we modify the function to read the
  // symbol's value from a global variable which we call the "globals
  // table".  The dynamic linker can then relocate the module by filling
  // out the globals table.
  //
  // For example, suppose we have a C library that contains this:
  //
  //   extern int imported_var;
  //
  //   int *get_imported_var() {
  //     return &imported_var;
  //   }
  //
  // We transform that code to the equivalent of this:
  //
  //   static void *__globals_table__[] = { &imported_var, ... };
  //
  //   int *get_imported_var() {
  //     return __globals_table__[0];
  //   }
  //
  // The relocation to "addr_of_imported_var" is then recorded by a later
  // part of the ConvertToPSO pass.
  //
  // The globals table does the same job as the Global Offset Table (GOT)
  // in ELF.  It is slightly different from the GOT because it is
  // implemented as a different level of abstraction.  In ELF, the GOT is a
  // linker feature.  Relocations can be relative to the GOT's base
  // address, and there can only be one GOT per ELF module.  The compiler
  // and assembler can generate GOT-relative relocations (when compiling
  // with "-fPIC"); the linker resolves these and generates the GOT.
  //
  // In contrast, in PNaCl the globals table is introduced at the level of
  // LLVM IR.  Unlike the GOT, the globals table is not special.  Nothing
  // needs to know about it outside this function.  (However, this would
  // change if we were to add support for lazy binding.)
  void buildGlobalsTable(Module &M) {
    // Search for all references to imported functions/variables by
    // functions.
    SmallVector<std::pair<Use *, unsigned>, 32> Refs;
    SmallVector<Constant *, 32> TableEntries;
    auto processGlobalValue = [&](GlobalValue &GV) {
      if (GV.isDeclaration()) {
        bool NeedsEntry = false;
        for (Use &U : GV.uses()) {
          if (isa<Instruction>(U.getUser())) {
            NeedsEntry = true;
            Refs.push_back(std::make_pair(&U, TableEntries.size()));
          }
        }
        if (NeedsEntry) {
          TableEntries.push_back(&GV);
        }
      }
    };
    for (Function &Func : M.functions()) {
      if (!Func.isIntrinsic())
        processGlobalValue(Func);
    }
    for (GlobalValue &Var : M.globals()) {
      processGlobalValue(Var);
    }

    if (TableEntries.empty())
      return;

    // Create a GlobalVariable for the globals table.
    Constant *TableData = ConstantStruct::getAnon(
        M.getContext(), TableEntries, true);
    auto TableVar = new GlobalVariable(
        M, TableData->getType(), false, GlobalValue::InternalLinkage,
        TableData, "__globals_table__");

    // Update use sites to load addresses from the globals table.
    for (auto &Ref : Refs) {
      Value *GV = Ref.first->get();
      Instruction *InsertPt = cast<Instruction>(Ref.first->getUser());
      Value *Indexes[] = {
        ConstantInt::get(M.getContext(), APInt(32, 0)),
        ConstantInt::get(M.getContext(), APInt(32, Ref.second)),
      };
      Value *TableEntryAddr = GetElementPtrInst::Create(
          TableData->getType(), TableVar, Indexes,
          GV->getName() + ".gt", InsertPt);
      Ref.first->set(new LoadInst(TableEntryAddr, GV->getName(), InsertPt));
    }
  }

  // Add a symbol to a specified StringTable, as well as a vector tracking
  // indices into the StringTable (and corresponding values).
  void addToStringTable(SmallVectorImpl<char> *StringTable, Type *IntPtrType,
                        SmallVectorImpl<Constant *> *NameOffsets,
                        const StringRef Name) {
    // Identify the offset in the StringTable that will contain the symbol name.
    NameOffsets->push_back(ConstantInt::get(IntPtrType, StringTable->size()));

    // Copy the name into the string table, along with the null terminator.
    StringTable->append(Name.begin(), Name.end());
    StringTable->push_back(0);
  }

  // handleLocalTlsVars() handles within-module references to thread-local (TLS)
  // variables -- i.e. references to TLS variables that are defined within
  // the module.
  //
  // Suppose we have the following access to a TLS variable:
  //
  //   static __thread int tls_var;
  //
  //   int *get_var() {
  //     return &tls_var;
  //   }
  //
  // We transform that code to the equivalent of this:
  //
  //   struct TLSBlockGetter {
  //     // Returns pointer to the module's TLS block for the current thread.
  //     void *(*func)(struct TLSBlockGetter *closure);
  //     // The dynamic linker can use "arg" to store an ID for the module.
  //     uintptr_t arg;
  //   } __tls_getter_closure;
  //
  //   int *get_var() {
  //     char *tls_base = __tls_getter_closure.func(&__tls_getter_closure);
  //     return (int *) (tls_base + OFFSET_FOR_TLS_VAR);
  //   }
  //
  // where OFFSET_FOR_TLS_VAR is a constant that is inlined into the code.
  //
  // Note that we only need one instance of TLSBlockGetter in the module.
  //
  // Exported TLS variables are added to the list of exported symbols (which are
  // processed later in the ConvertToPSO pass), and exported as the offset from
  // the base of the TLS block for a given module. This information is used by
  // the dynamic linker when resolving symbols.
  void handleLocalTlsVars(
      Module &M, SmallVectorImpl<char> *StringTable, Type *PtrType,
      Type *IntPtrType, SmallVectorImpl<TrackingVH<Constant>> *PsoRootTlsFields,
      SmallVectorImpl<SymbolTableEntry> *ExportedSymbolTableVector) {
    TlsTemplate Templ;
    buildTlsTemplate(M, &Templ);

    // Construct the TLSBlockGetter struct type and its function type.
    StructType *ClosureType =
        StructType::create(M.getContext(), "TLSBlockGetter");
    Type *Args[] = { ClosureType->getPointerTo() };
    FunctionType *FuncType = FunctionType::get(
        IntPtrType, Args, /*isVarArg=*/false);
    Type *ClosureTypeFields[] = {
      FuncType->getPointerTo(),
      IntPtrType,
    };
    ClosureType->setBody(ClosureTypeFields);

    Constant *TemplateDataVar = new GlobalVariable(
        M, Templ.Data->getType(), /*isConstant=*/true,
        GlobalValue::InternalLinkage, Templ.Data, "__tls_template");
    Constant *Closure = new GlobalVariable(
        M, ClosureType, /*isConstant=*/false, GlobalValue::InternalLinkage,
        Constant::getNullValue(ClosureType), "__tls_getter_closure");

    PsoRootTlsFields->push_back(TemplateDataVar);
    PsoRootTlsFields->push_back(ConstantInt::get(IntPtrType, Templ.DataSize));
    PsoRootTlsFields->push_back(ConstantInt::get(IntPtrType, Templ.TotalSize));
    PsoRootTlsFields->push_back(ConstantInt::get(IntPtrType, Templ.Alignment));
    PsoRootTlsFields->push_back(Closure);

    // Rewrite accesses to the TLS variables.
    for (auto &VarInfo : Templ.TlsVars) {
      GlobalVariable *Var = VarInfo.TlsVar;

      // Delete unused ConstantExprs that reference Var, because the code
      // that follows assumes none remain.  ExpandConstantExprPass is one
      // pass that is known to leave dead ConstantExprs behind, but other
      // passes might too.
      Var->removeDeadConstantUsers();

      while (!Var->use_empty()) {
        Use *U = &*Var->use_begin();
        Instruction *InsertPt = PhiSafeInsertPt(U);

        Value *Indexes[] = {
          ConstantInt::get(M.getContext(), APInt(32, 0)),
          ConstantInt::get(M.getContext(), APInt(32, 0)),
        };
        Value *FuncField = GetElementPtrInst::Create(
            ClosureType, Closure, Indexes, "tls_getter_func_field", InsertPt);
        Value *CallArgs[] = { Closure };
        Value *Func = new LoadInst(FuncField, "tls_getter_func", InsertPt);
        Value *Tp = CallInst::Create(Func, CallArgs, "tls_base", InsertPt);
        Value *TlsField = BinaryOperator::Create(
            Instruction::Add, Tp, ConstantInt::get(IntPtrType, VarInfo.Offset),
            Var->getName(), InsertPt);
        Value *Ptr = new IntToPtrInst(TlsField, Var->getType(),
                                      Var->getName() + ".ptr", InsertPt);
        PhiSafeReplaceUses(U, Ptr);
      }

      // For exported TLS variables, only export a constant offset from the base
      // of the TLS block. This information can be processed by the dynamic
      // linker to find the actual thread-local symbol at runtime.
      // We explicitly copy Var's name into the ExportedSymbolTable since
      // Constants cannot have names in LLVM.
      if (Var->getLinkage() == GlobalValue::ExternalLinkage) {
        Constant *ExportOffset =
            Constant::getIntegerValue(PtrType, APInt(32, VarInfo.Offset));
        ExportedSymbolTableVector->push_back(SymbolTableEntry(Var->getName(),
                                                              ExportOffset));
      }
      Var->eraseFromParent();
    }
  }

  // handleImportedTlsVars() handles references to extern thread-local (TLS)
  // variables -- i.e. references to TLS variables that are defined in other
  // modules.
  //
  // Suppose we have the following access to a TLS variable:
  //
  //   extern __thread int tls_var_extern;
  //
  //   int *get_var() {
  //     return &tls_var_extern;
  //   }
  //
  // We transform that code to the equivalent of this:
  //
  //   struct TLSVarGetter {
  //     // Returns pointer to the module's TLS block for the current thread.
  //     void *(*func)(struct TLSVarGetter *closure);
  //     // The dynamic linker can use "arg1" to store an ID for the module.
  //     uintptr_t arg1;
  //     // The dynamic linker can use "arg2" to store the offset of the
  //     // variable in the module's TLS block.
  //     uintptr_t arg2;
  //   } __tls_var_extern;
  //
  //   int *get_var() {
  //     return __tls_var_extern.func(&__tls_var_extern);
  //   }
  //
  // All references to the imported TLS variable are replaced with a
  // "TLSVarGetter" closure, which is filled in by the dynamic linker so that it
  // can return the actual address of the imported variable at runtime.
  //
  // Note that we need one instance of TLSVarGetter per extern TLS variable in
  // the module.
  void handleImportedTlsVars(
      Module &M, SmallVectorImpl<char> *StringTable, Type *PtrType,
      Type *IntPtrType,
      SmallVectorImpl<TrackingVH<Constant>> *PsoRootTlsFields) {
    StructType *TLSVarClosureType =
        StructType::create(M.getContext(), "TLSVarGetter");
    Type *TLSVarClosureArgs[] = { TLSVarClosureType->getPointerTo() };
    FunctionType *TLSVarClosureFuncType = FunctionType::get(
          IntPtrType, TLSVarClosureArgs, /*isVarArg=*/false);
    Type *TLSVarClosureTypeFields[] = {
      TLSVarClosureFuncType->getPointerTo(),
      IntPtrType,
      IntPtrType,
    };
    TLSVarClosureType->setBody(TLSVarClosureTypeFields);

    // Imported TLS GlobalVariables are stored separately, since the number of
    // imported TLS variables must be calculated for ImportTLSArrayType, and the
    // intermediate vector can prevent re-iterating over all globals.
    std::vector<GlobalVariable *> ImportedTLSVars;
    for (GlobalVariable &Var : M.globals()) {
      if (Var.isThreadLocal() && !Var.hasInitializer()) {
        ImportedTLSVars.push_back(&Var);
      }
    }

    // Create a TLSVarClosure for each imported TLS variable.
    // These imported TLS variables will not exist in the structure returned
    // from "buildTlsTemplate", since they do not have initializers.
    ArrayType *ImportTLSArrayType =
        ArrayType::get(TLSVarClosureType, ImportedTLSVars.size());
    GlobalVariable *ImportTLSArray = new GlobalVariable(
        M, ImportTLSArrayType, false, GlobalValue::InternalLinkage,
        Constant::getNullValue(ImportTLSArrayType), "import_tls_getter_array");
    SmallVector<Constant *, 32> ImportTLSNames;

    for (GlobalVariable *Var : ImportedTLSVars) {
      // Delete unused ConstantExprs that reference Var, because the code
      // that follows assumes none remain.  ExpandConstantExprPass is one
      // pass that is known to leave dead ConstantExprs behind, but other
      // passes might too.
      Var->removeDeadConstantUsers();

      while (!Var->use_empty()) {
        // For each use of the imported TLS variable, replace it with function
        // call to the TLSVarGetter closure.
        Use *U = &*Var->use_begin();
        Instruction *InsertPt = PhiSafeInsertPt(U);

        // ImportedValue = imported_tls_var_desc.getter(&imported_tls_var_desc);
        Value *ArrayIndexes[] = {
          ConstantInt::get(M.getContext(), APInt(32, 0)),
          ConstantInt::get(M.getContext(), APInt(32, ImportTLSNames.size())),
        };
        Value *TLSVarClosure = GetElementPtrInst::Create(
            ImportTLSArrayType, ImportTLSArray, ArrayIndexes,
            Var->getName() + ".tls_imported_array_access", InsertPt);
        Value *StructIndexes[] = {
          ConstantInt::get(M.getContext(), APInt(32, 0)),
          ConstantInt::get(M.getContext(), APInt(32, 0)),
        };
        Value *FuncField = GetElementPtrInst::Create(
            TLSVarClosureType, TLSVarClosure, StructIndexes,
            Var->getName() + ".tls_imported_struct_access", InsertPt);
        Value *CallArgs[] = { TLSVarClosure };
        Value *Func = new LoadInst(FuncField,
                                   Var->getName() + ".tls_imported_getter_func",
                                   InsertPt);
        Value *ImportedValue = CallInst::Create(Func, CallArgs, Var->getName(),
                                                InsertPt);
        Value *Ptr = new IntToPtrInst(ImportedValue, Var->getType(),
                                      Var->getName() + ".ptr", InsertPt);
        PhiSafeReplaceUses(U, Ptr);
      }
      // Add the imported TLS variable name to the symbol table.
      addToStringTable(StringTable, IntPtrType, &ImportTLSNames,
                       Var->getName());
      Var->eraseFromParent();
    }

    PsoRootTlsFields->push_back(ImportTLSArray);
    PsoRootTlsFields->push_back(
        createArray(M, "import_tls_names", &ImportTLSNames, IntPtrType)),
    PsoRootTlsFields->push_back(
        ConstantInt::get(IntPtrType, ImportedTLSVars.size()));
  }
}

char ConvertToPSO::ID = 0;
INITIALIZE_PASS(ConvertToPSO, "convert-to-pso",
                "Convert module to a PNaCl portable shared object (PSO)",
                false, false)

bool ConvertToPSO::runOnModule(Module &M) {
  LLVMContext &C = M.getContext();
  DataLayout DL(&M);
  Type *PtrType = Type::getInt8Ty(C)->getPointerTo();
  Type *IntPtrType = DL.getIntPtrType(C);

  // Both handleTlsVars() and buildGlobalsTable() need to replace some uses
  // of GlobalValues with instruction sequences, but that only works if
  // functions don't contain ConstantExprs referencing those GlobalValues,
  // because we can't modify a ConstantExpr to refer to an instruction.  To
  // address this, we first convert all ConstantExprs inside functions into
  // instructions by running the ExpandConstantExpr pass.
  FunctionPass *ConstExprPass = createExpandConstantExprPass();
  for (Function &Func : M.functions())
    ConstExprPass->runOnFunction(Func);
  delete ConstExprPass;

  // A table of strings which contains all imported and exported symbol names.
  SmallString<1024> StringTable;

  // This acts roughly like the ".dynsym" section of an ELF file.
  // It contains all symbols which will be exported.
  SmallVector<SymbolTableEntry, 32> ExportedSymbolTableVector;

  // Enters the name of a symbol into the string table, and record
  // the index at which the symbol is stored in the list of names.
  auto createSymbol = [&](SmallVectorImpl<Constant *> *NameOffsets,
                          SmallVectorImpl<Constant *> *ValuePtrs,
                          const StringRef Name, Constant *Addr) {
    // Identify the symbol's address (for exports) or the address which should
    // be updated to include the symbol (for imports).
    ValuePtrs->push_back(ConstantExpr::getBitCast(Addr, PtrType));

    addToStringTable(&StringTable, IntPtrType, NameOffsets, Name);
  };

  // We need to wrap these Constants with TrackingVH<> because our later
  // call to FlattenGlobals will recreate some of them with different
  // types.
  SmallVector<TrackingVH<Constant>, 5> PsoRootLocalTlsFields;
  SmallVector<TrackingVH<Constant>, 3> PsoRootImportedTlsFields;
  handleLocalTlsVars(M, &StringTable, PtrType, IntPtrType,
                     &PsoRootLocalTlsFields, &ExportedSymbolTableVector);
  handleImportedTlsVars(M, &StringTable, PtrType, IntPtrType,
                        &PsoRootImportedTlsFields);

  buildGlobalsTable(M);

  // In order to simplify the task of processing relocations inside
  // GlobalVariables' initializers, we first run the FlattenGlobals pass to
  // reduce initializers to a simple normal form.  This reduces the number
  // of cases we need to handle, and it allows us to iterate over the
  // initializers instead of needing to recurse.
  ModulePass *Pass = createFlattenGlobalsPass();
  Pass->runOnModule(M);
  delete Pass;

  // Process imports.
  SmallVector<Constant *, 32> ImportPtrs;
  // Indexes into the StringTable for the names of exported symbols.
  SmallVector<Constant *, 32> ImportNames;
  for (GlobalVariable &Var : M.globals()) {
    if (!Var.hasInitializer())
      continue;
    Constant *Init = Var.getInitializer();
    if (auto CS = dyn_cast<ConstantStruct>(Init)) {
      // The initializer is a CompoundElement (i.e. a struct containing
      // SimpleElements).
      SmallVector<Constant *, 32> Elements;
      bool Modified = false;

      for (unsigned I = 0; I < CS->getNumOperands(); ++I) {
        Constant *Element = CS->getOperand(I);
        uint64_t Addend;
        if (auto GV = getReference(Element, &Addend)) {
          // Calculate the address that needs relocating.
          Value *Indexes[] = {
            ConstantInt::get(C, APInt(32, 0)),
            ConstantInt::get(C, APInt(32, I)),
          };
          Constant *Addr = ConstantExpr::getGetElementPtr(
              Init->getType(), &Var, Indexes);
          createSymbol(&ImportNames, &ImportPtrs, GV->getName(), Addr);
          // Replace the original reference with the addend value.
          Element = ConstantInt::get(Element->getType(), Addend);
          Modified = true;
        }
        Elements.push_back(Element);
      }

      if (Modified) {
        // This global variable will need to be relocated at runtime, so it
        // should not be in read-only memory.
        Var.setConstant(false);
        // Note that the resulting initializer will not follow
        // FlattenGlobals' normal form, because it will contain i32s rather
        // than i8 arrays.  However, the later pass of FlattenGlobals will
        // restore the normal form.
        Var.setInitializer(ConstantStruct::getAnon(C, Elements, true));
      }
    } else {
      // The initializer is a single SimpleElement.
      uint64_t Addend;
      if (auto GV = getReference(Init, &Addend)) {
        createSymbol(&ImportNames, &ImportPtrs, GV->getName(), &Var);
        // This global variable will need to be relocated at runtime, so it
        // should not be in read-only memory.
        Var.setConstant(false);
        // Replace the original reference with the addend value.
        Var.setInitializer(ConstantInt::get(Init->getType(), Addend));
      }
    }
  }

  // Process exports.
  SmallVector<Constant *, 32> ExportPtrs;
  // Indexes into the StringTable for the names of exported symbols.
  SmallVector<Constant *, 32> ExportNames;

  auto processGlobalValue = [&](GlobalValue &GV) {
    if (GV.isDeclaration()) {
      // Aside from intrinsics, we should have handled any imported
      // references already.
      if (auto Func = dyn_cast<Function>(&GV)) {
        if (Func->isIntrinsic())
          return;
      }
      GV.removeDeadConstantUsers();
      assert(GV.use_empty());
      GV.eraseFromParent();
      return;
    }

    if (GV.getLinkage() != GlobalValue::ExternalLinkage)
      return;

    // Actually store the pointer to be exported.
    ExportedSymbolTableVector.push_back(SymbolTableEntry(GV.getName(), &GV));
    GV.setLinkage(GlobalValue::InternalLinkage);
  };

  for (auto Iter = M.begin(); Iter != M.end(); ) {
    processGlobalValue(*Iter++);
  }
  for (auto Iter = M.global_begin(); Iter != M.global_end(); ) {
    processGlobalValue(*Iter++);
  }
  for (auto Iter = M.alias_begin(); Iter != M.alias_end(); ) {
    processGlobalValue(*Iter++);
  }

  // The following section uses the ExportedSymbolTableVector to generate a hash
  // table, which, embeded in PSL Root, can be used to quickly look up symbols
  // based on a string name.
  //
  // The hash table is based on the GNU ELF hash section
  // (https://blogs.oracle.com/ali/entry/gnu_hash_elf_sections).
  //
  // Using the hash table requires the following function be known:
  //   uint32_t hashString(const char *str)
  //
  // The hash table contains the following fields:
  //   size_t NumBuckets
  //   int32_t *Buckets[0 ... NumBuckets]
  //   uint32_t *HashChains[0 ... NumChainEntries]
  // Where NumChainEntries is known to be the number of exported symbols.
  //
  // The hash table requires that the list of ExportNames is sorted by
  // "hashString(export_symbol) % NumBuckets".
  //
  // Given an input string, Str, a lookup is done as follows:
  // 1) H = hashString(Str) is calculated.
  // 2) BucketIndex = H % NumBuckets is calculated, as an index into the list of
  //    buckets.
  // 3) BucketValue = Buckets[BucketIndex] is calculated.
  //    BucketValue will be -1 if there are no exported symbols such that
  //      hashString(symbol.name) % NumBuckets = BucketIndex.
  //    BucketValue will be an index into the HashChains array if there is at
  //      least one symbol where hashString(symbol.name) % NumBuckets =
  //      BucketIndex.
  // 4) If BucketValue != -1, then BucketValue corresponds with an index to the
  //    start of a chain, identified as "ChainIndex".
  // 5) ChainIndex has a double meaning.
  //    Firstly, ChainIndex itself is an index into "ExportNames" (taking
  //      advantage of the sorting requirement stated earlier).
  //    Secondly, ChainValue = HashChains[ChainIndex] can be calculated.
  //    ChainValue also has a double meaning:
  //      The bottom bit (ChainValue & 1):
  //        This bit indicates if ExportNames[ChainIndex] is the last symbol
  //        with a name such that:
  //        BucketIndex == hashString(ExportNames[ChainIndex]) % NumBuckets
  //        In other words, this bit is 1 if ChainIndex is the end of a chain.
  //      The top 31 bits (ChainValue & ~1):
  //        The top 31 bits of ChainValue cache the hash of the corresonding
  //        symbol in export names:
  //        ChainValue = hashString(ExportNames[ChainIndex]) & ~1
  //        This hash can be used to quickly compare with "H".
  // 6) For each entry in the chain, (ChainValue & ~1) can be compared with
  //    (H & ~1) to quickly identify if "Str" matches the corresponding symbol
  //    at ExportNames[ChainIndex]. If the hashes match, the full strings are
  //    compared. If they do not, ChainIndex is incremented, and step (6) is
  //    repeated (unless the ChainIndex is the end of a chain, indicated by
  //    ChainValue & 1).

  const size_t NumChainEntries = ExportedSymbolTableVector.size();
  const size_t AverageChainLength = 4;
  const size_t NumBuckets = (NumChainEntries + AverageChainLength - 1)
      / AverageChainLength;

  // The SymbolTable must be sorted by hash(symbol name) % number of buckets
  // to allow quick access from the hash table.
  // Sort the table (as a vector), and then iterate through the symbols, adding
  // their names and values to the appropriate variable in the PSLRoot.
  auto sortStringTable = [&](const SymbolTableEntry &A,
                             const SymbolTableEntry &B) {
    return (A.Hash % NumBuckets) <
           (B.Hash % NumBuckets);
  };

  std::sort(ExportedSymbolTableVector.begin(), ExportedSymbolTableVector.end(),
            sortStringTable);

  SmallVector<uint32_t, 32> HashBuckets;
  HashBuckets.assign(NumBuckets, -1);

  SmallVector<uint32_t, 32> HashChains;
  HashChains.reserve(ExportPtrs.size());

  // A bloom filter is used to quickly reject queries for exported symbols which
  // do not exist within a module.
  //
  // The bloom filter is composed of |MaskWords| number of words, each 32 bits
  // wide.
  //
  // It uses k=2, or "two independent hash functions for each symbol". This
  // means that two separate hashes, |H1| and |H2| are calculated. Each of these
  // hashes corresponds to a single MaskWord and MaskBit pair. This lets a
  // single bit of the bloom filter be turned on based on the inclusion of a
  // hash.
  //
  // These hashes are calculated as follows:
  //    H1 = Hash(name);
  //    H2 = H1 >> Shift2;
  //
  // Where Shift2 is the number of right-shifts to generate H2 (from H1).
  //
  // Inspired by the GNU hash section, the bloom filter's performance improves
  // when a single MaskWord is used for the pair of hashes, rather than two
  // separate MaskWords.
  //
  //    MaskWord = (H1 / 32) % MaskWords;
  //    Bitmask = (1 << (H1 % 32)) | (1 << (H2 % 32));
  //
  // As a bit-fiddling trick, |MaskWords| must be a power of 2.
  // This lets us avoid the modulus operation when calculating MaskWord:
  //    MaskWord = (H1 / 32) & (MaskWords - 1);

  size_t ExportSymCount = ExportedSymbolTableVector.size();
  size_t MaskBitsLog2 = Log2_32(ExportSymCount) + 1;
  const int Log2BitsPerWord = 5;

  // This heuristic for calculating Shift2 and MaskWords matches the heuristic
  // used in both BFD ld and gold.
  if (MaskBitsLog2 < 3)
    MaskBitsLog2 = Log2BitsPerWord;
  else if ((1 << (MaskBitsLog2 - 2)) & ExportSymCount)
    // If the number of exported symbols is almost high enough to trigger an
    // increased value of "MaskBitsLog2", then add one anyway, to decrease the
    // likeliness of hash collisions. For example, for values of ExportSymCount
    // in the range of 32 to 47, the conditional will evaluate to "false". For
    // the range 48 to 63, it will evaluate to "true".
    MaskBitsLog2 += 3;
  else
    MaskBitsLog2 += 2;

  const size_t Shift2 = MaskBitsLog2;
  assert(Shift2 >= Log2BitsPerWord);
  const size_t MaskWords = 1U << (Shift2 - Log2BitsPerWord);
  SmallVector<uint32_t, 32> BloomFilter;
  BloomFilter.assign(MaskWords, 0);

  uint32_t PrevBucketNum = -1;
  for (size_t Index = 0; Index < ExportedSymbolTableVector.size(); ++Index) {
    const SymbolTableEntry *Element = &ExportedSymbolTableVector[Index];
    const uint32_t HashValue = Element->Hash;

    // Update bloom filter.
    const uint32_t HashValue2 = Element->Hash >> Shift2;
    uint32_t WordNum = (HashValue / 32) % MaskWords;
    uint32_t Bitmask = (1 << (HashValue % 32)) | (1 << (HashValue2 % 32));
    BloomFilter[WordNum] |= Bitmask;

    // The bottom bit of the chain value is reserved to identify if the element
    // is the "end of the chain" for the given (hash(name) % numbuckets) entry.
    uint32_t ChainValue = HashValue & ~1;
    // The final entry in the chain list should be marked as "end of chain".
    if (Index == ExportedSymbolTableVector.size() - 1)
      ChainValue |= 1;
    HashChains.push_back(ChainValue);

    uint32_t BucketNum = HashValue % NumBuckets;
    if (PrevBucketNum != BucketNum) {
      // We are starting a new hash chain.
      if (Index != 0) {
        // Mark the end of the previous hash chain.
        HashChains[Index - 1] |= 1;
      }
      // Record a pointer to the start of the new hash chain.
      HashBuckets[BucketNum] = Index;
      PrevBucketNum = BucketNum;
    }

    createSymbol(&ExportNames, &ExportPtrs, Element->Name, Element->Value);
  }

  // This lets us remove the "NumChainEntries" field from the PsoRoot.
  assert(NumChainEntries == ExportPtrs.size() && "Malformed export hash table");

  // Set up string of exported names.
  Constant *StringTableArray = ConstantDataArray::getString(
      C, StringRef(StringTable.data(), StringTable.size()), false);
  Constant *StringTableVar = new GlobalVariable(
      M, StringTableArray->getType(), true, GlobalValue::InternalLinkage,
      StringTableArray, "string_table");

  // Set up string of PLL dependencies.
  SmallString<1024> DependenciesList;
  for (auto dependency : LibraryDependencies) {
    DependenciesList.append(dependency);
    DependenciesList.push_back(0);
  }
  Constant *DependenciesListArray = ConstantDataArray::getString(
      C, StringRef(DependenciesList.data(), DependenciesList.size()), false);
  Constant *DependenciesListVar = new GlobalVariable(
      M, DependenciesListArray->getType(), true, GlobalValue::InternalLinkage,
      DependenciesListArray, "dependencies_list");

  SmallVector<Constant *, 32> PsoRoot = {
    ConstantInt::get(IntPtrType, PSOFormatVersion),

    // String Table
    StringTableVar,

    // Exports
    createArray(M, "export_ptrs", &ExportPtrs, PtrType),
    createArray(M, "export_names", &ExportNames, IntPtrType),
    ConstantInt::get(IntPtrType, ExportPtrs.size()),

    // Imports
    createArray(M, "import_ptrs", &ImportPtrs, PtrType),
    createArray(M, "import_names", &ImportNames, IntPtrType),
    ConstantInt::get(IntPtrType, ImportPtrs.size()),

    // Hash Table (for quick string lookup of exports)
    ConstantInt::get(IntPtrType, NumBuckets),
    createDataArray(M, "hash_buckets", &HashBuckets),
    createDataArray(M, "hash_chains", &HashChains),

    // Bloom Filter (for quick string lookup rejection of exports)
    ConstantInt::get(IntPtrType, MaskWords - 1),
    ConstantInt::get(IntPtrType, Shift2),
    createDataArray(M, "bloom_filter", &BloomFilter),
  };
  for (auto FieldVal : PsoRootLocalTlsFields)
    PsoRoot.push_back(FieldVal);

  // Dependencies List
  PsoRoot.push_back(ConstantInt::get(IntPtrType, LibraryDependencies.size()));
  PsoRoot.push_back(DependenciesListVar);

  // TODO(smklein): Combine both PsoRoots into "PsoRootTlsFields", and reorder
  // fields within the PLL root.
  for (auto FieldVal : PsoRootImportedTlsFields)
    PsoRoot.push_back(FieldVal);

  Constant *PsoRootConst = ConstantStruct::getAnon(PsoRoot);
  new GlobalVariable(
      M, PsoRootConst->getType(), true, GlobalValue::ExternalLinkage,
      PsoRootConst, "__pnacl_pso_root");

  // As soon as we have finished exporting aliases, we can resolve them.
  ModulePass *AliasPass = createResolveAliasesPass();
  AliasPass->runOnModule(M);
  delete AliasPass;

  return true;
}

ModulePass *llvm::createConvertToPSOPass() {
  return new ConvertToPSO();
}
