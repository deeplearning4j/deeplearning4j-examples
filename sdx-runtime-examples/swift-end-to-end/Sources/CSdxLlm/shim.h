#ifndef CSDXLLM_SHIM_H
#define CSDXLLM_SHIM_H

/*
 * Resolve sdx_llm_c.h across two supported layouts:
 *  - unpacked AOT SDK:   <sdk-root>/include/sdx_llm_c.h
 *    Set SDX_LLM_INCLUDE to that directory, or point -Xcc -I<dir> at it.
 *  - source tree:        nd4j/sdx-aot/include/sdx_llm_c.h
 *    This package lives at sdx-runtime-examples/swift-end-to-end/.
 *
 * The module.modulemap links "sdx_llm" (the shared library stem);
 * supply the library via -Xlinker -L<sdk>/lib at build time.
 */
#if __has_include("sdx_llm_c.h")
  /* SDK include/ dir already on the search path (-Xcc -I<sdk>/include). */
  #include "sdx_llm_c.h"
#elif __has_include("../../../../nd4j/sdx-aot/include/sdx_llm_c.h")
  /* Source-tree path: examples repo sits next to deeplearning4j checkout. */
  #include "../../../../nd4j/sdx-aot/include/sdx_llm_c.h"
#else
  #error "sdx_llm_c.h not found. Pass -Xcc -I<sdk>/include to swift build."
#endif

#endif /* CSDXLLM_SHIM_H */
