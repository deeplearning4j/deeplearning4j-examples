plugins {
    kotlin("jvm") version "1.9.24"
    application
}

group = "org.nd4j.examples"
version = "0.1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation("net.java.dev.jna:jna:5.14.0")
}

// Wrapper sources are compiled straight into this example. The default paths
// assume a sibling `deeplearning4j` checkout next to this examples repository;
// when building against an unpacked SDK package, override with:
//   gradle run -PsdxWrappersDir=/path/to/sdk/wrappers
val sdxWrappersDir: String = (findProperty("sdxWrappersDir") as String?)
    ?: "../../../deeplearning4j/libnd4j/include/dsp/runtime/bindings"
val sdxJavaSrcDir: String = (findProperty("sdxJavaSrcDir") as String?)
    ?: "../../../deeplearning4j/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx/src/main/java"

sourceSets {
    main {
        java {
            // The JNA-based Java wrapper (org.nd4j.dsp.runtime.SdxRuntime).
            // First path: SDK package layout (wrappers/java/src/main/java);
            // second: the canonical nd4j-sdx module in a sibling checkout.
            // Gradle silently skips whichever does not exist.
            srcDir("$sdxWrappersDir/java/src/main/java")
            srcDir(sdxJavaSrcDir)
        }
        kotlin {
            // The Kotlin facade (org.nd4j.dsp.runtime.KotlinSdxRuntime).
            srcDir("$sdxWrappersDir/kotlin/src/main/kotlin")
        }
    }
}

kotlin {
    jvmToolchain(11)
}

application {
    mainClass.set("org.nd4j.examples.sdx.EndToEndKt")
}

// ── Secondary entry: LLM end-to-end example ───────────────────────────────────
// Usage:
//   export SDX_LLM_AOT_HOME=/tmp/sdx-cpu-v8
//   export SDX_NATIVE_LIB_DIR=$SDX_LLM_AOT_HOME/lib
//   gradle llmRun --args="path/to/model.gguf path/to/tokenizer.json"
tasks.register<JavaExec>("llmRun") {
    group = "application"
    description = "Run the SDX LLM end-to-end example (requires libsdx_llm.so)"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("org.nd4j.examples.sdx.LlmEndToEndKt")
}
