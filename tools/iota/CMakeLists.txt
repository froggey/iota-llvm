set(LLVM_LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  Core
  IRReader
  ScalarOpts
  SelectionDAG
  Support
  )

add_llvm_tool(iota
  iota.cpp
  )
