set(LLVM_LINK_COMPONENTS
  AsmParser
  BitReader
  BitWriter
  NaClBitAnalysis
  NaClBitTestUtils
  NaClBitReader
  NaClBitWriter
  Core
  Support
  )

add_llvm_unittest(BitcodeTests
  BitReaderTest.cpp
  BitstreamReaderTest.cpp
  NaClAbbrevTrieTest.cpp
  NaClBitReaderTest.cpp
  NaClBitstreamReaderTest.cpp
  NaClCompressTests.cpp
  NaClMungedBitcodeTest.cpp
  NaClMungedIoTest.cpp
  NaClMungeTest.cpp
  NaClMungeWriteErrorTests.cpp
  NaClObjDumpTest.cpp
  NaClObjDumpTypesTest.cpp
  NaClParseTypesTest.cpp
  NaClParseInstsTest.cpp
  NaClTextFormatterTest.cpp
  )
