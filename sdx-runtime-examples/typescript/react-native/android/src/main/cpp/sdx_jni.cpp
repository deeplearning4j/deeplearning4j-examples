/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

// JNI shim between the React Native Kotlin module and the SDX C ABI
// (dsp_runtime_c.h). All sdx* calls live here; the Kotlin side only marshals
// React Native types. Handles cross JNI as jlong (the raw sdx pointers); a
// single process-wide sdx_runtime_t backs every model.

#include <dsp/runtime/dsp_runtime_c.h>

#include <jni.h>

#include <mutex>
#include <string>
#include <vector>

namespace {

sdx_runtime_t* gRuntime = nullptr;
std::mutex gRuntimeMutex;
std::string gLastError;

sdx_runtime_t* runtime() {
  std::lock_guard<std::mutex> lock(gRuntimeMutex);
  if (gRuntime == nullptr) {
    sdx_runtime_options_t options{};
    options.struct_size = sizeof(options);
    if (sdxCreateRuntime(&options, &gRuntime) != SDX_STATUS_OK) {
      gRuntime = nullptr;
    }
  }
  return gRuntime;
}

void recordError(const char* op, sdx_status_t status) {
  const char* detail = gRuntime != nullptr ? sdxGetLastError(gRuntime) : "";
  gLastError = std::string(op) + " failed: status=" + std::to_string(status) +
               (detail != nullptr && detail[0] != '\0' ? std::string(", error=") + detail : "");
}

std::string toStdString(JNIEnv* env, jstring value) {
  const char* chars = env->GetStringUTFChars(value, nullptr);
  std::string result(chars != nullptr ? chars : "");
  env->ReleaseStringUTFChars(value, chars);
  return result;
}

}  // namespace

extern "C" {

JNIEXPORT jint JNICALL
Java_org_nd4j_sdx_rn_SdxRuntimeModule_nativeAbiVersion(JNIEnv*, jobject) {
  return sdxGetRuntimeAbiVersion();
}

JNIEXPORT jstring JNICALL
Java_org_nd4j_sdx_rn_SdxRuntimeModule_nativeLastError(JNIEnv* env, jobject) {
  return env->NewStringUTF(gLastError.c_str());
}

JNIEXPORT jlong JNICALL
Java_org_nd4j_sdx_rn_SdxRuntimeModule_nativeLoadModel(JNIEnv* env, jobject, jstring path) {
  sdx_runtime_t* rt = runtime();
  if (rt == nullptr) {
    gLastError = "sdxCreateRuntime failed";
    return 0;
  }
  const std::string bundlePath = toStdString(env, path);
  sdx_model_options_t options{};
  options.struct_size = sizeof(options);
  sdx_model_t* model = nullptr;
  const sdx_status_t status = sdxLoadBundle(rt, bundlePath.c_str(), &options, &model);
  if (status != SDX_STATUS_OK) {
    recordError("sdxLoadBundle", status);
    return 0;
  }
  return reinterpret_cast<jlong>(model);
}

JNIEXPORT void JNICALL
Java_org_nd4j_sdx_rn_SdxRuntimeModule_nativeUnloadModel(JNIEnv*, jobject, jlong model) {
  sdxUnloadModel(reinterpret_cast<sdx_model_t*>(model));
}

JNIEXPORT jlong JNICALL
Java_org_nd4j_sdx_rn_SdxRuntimeModule_nativeCreateContext(
    JNIEnv* env, jobject, jlong model, jobjectArray outputs) {
  const jsize count = env->GetArrayLength(outputs);
  std::vector<std::string> names;
  std::vector<const char*> namePtrs;
  names.reserve(count);
  namePtrs.reserve(count);
  for (jsize i = 0; i < count; i++) {
    auto name = static_cast<jstring>(env->GetObjectArrayElement(outputs, i));
    names.push_back(toStdString(env, name));
    env->DeleteLocalRef(name);
  }
  for (auto& name : names) namePtrs.push_back(name.c_str());

  sdx_context_t* context = nullptr;
  const sdx_status_t status = sdxCreateContext(
      reinterpret_cast<sdx_model_t*>(model),
      namePtrs.empty() ? nullptr : namePtrs.data(),
      static_cast<int32_t>(namePtrs.size()), &context);
  if (status != SDX_STATUS_OK) {
    recordError("sdxCreateContext", status);
    return 0;
  }
  return reinterpret_cast<jlong>(context);
}

JNIEXPORT void JNICALL
Java_org_nd4j_sdx_rn_SdxRuntimeModule_nativeDestroyContext(JNIEnv*, jobject, jlong context) {
  sdxDestroyContext(reinterpret_cast<sdx_context_t*>(context));
}

JNIEXPORT jobjectArray JNICALL
Java_org_nd4j_sdx_rn_SdxRuntimeModule_nativeGetInputNames(JNIEnv* env, jobject, jlong context) {
  auto* ctx = reinterpret_cast<sdx_context_t*>(context);
  const int32_t numInputs = sdxGetNumInputs(ctx);
  jobjectArray result = env->NewObjectArray(
      numInputs < 0 ? 0 : numInputs, env->FindClass("java/lang/String"), nullptr);
  for (int32_t i = 0; i < numInputs; i++) {
    const char* name = sdxGetInputName(ctx, i);
    env->SetObjectArrayElement(result, i, env->NewStringUTF(name != nullptr ? name : ""));
  }
  return result;
}

JNIEXPORT jint JNICALL
Java_org_nd4j_sdx_rn_SdxRuntimeModule_nativeGetNumOutputs(JNIEnv*, jobject, jlong context) {
  return sdxGetNumOutputs(reinterpret_cast<sdx_context_t*>(context));
}

JNIEXPORT jint JNICALL
Java_org_nd4j_sdx_rn_SdxRuntimeModule_nativeMarkInputVariable(
    JNIEnv*, jobject, jlong context, jint index) {
  return sdxMarkInputVariable(reinterpret_cast<sdx_context_t*>(context), index);
}

JNIEXPORT jint JNICALL
Java_org_nd4j_sdx_rn_SdxRuntimeModule_nativeMarkInputPlaceholder(
    JNIEnv*, jobject, jlong context, jint index) {
  return sdxMarkInputPlaceholder(reinterpret_cast<sdx_context_t*>(context), index);
}

JNIEXPORT jint JNICALL
Java_org_nd4j_sdx_rn_SdxRuntimeModule_nativeFreezeShapes(JNIEnv*, jobject, jlong context) {
  return sdxFreezeShapes(reinterpret_cast<sdx_context_t*>(context));
}

JNIEXPORT jint JNICALL
Java_org_nd4j_sdx_rn_SdxRuntimeModule_nativeGetPlanPhase(JNIEnv*, jobject, jlong context) {
  return sdxGetPlanPhase(reinterpret_cast<sdx_context_t*>(context));
}

JNIEXPORT jint JNICALL
Java_org_nd4j_sdx_rn_SdxRuntimeModule_nativeGetExecutionCount(JNIEnv*, jobject, jlong context) {
  return sdxGetExecutionCount(reinterpret_cast<sdx_context_t*>(context));
}

/**
 * Executes the plan. Inputs/outputs are float32 host tensors: data as
 * float[][] and shapes as long[][], positionally matching the plan's input
 * contract and the requested outputs. Returns float[][] outputs, or null on
 * failure (nativeLastError() has the detail).
 */
JNIEXPORT jobjectArray JNICALL
Java_org_nd4j_sdx_rn_SdxRuntimeModule_nativeRun(
    JNIEnv* env, jobject, jlong context,
    jobjectArray inputData, jobjectArray inputShapes, jobjectArray outputShapes) {
  auto* ctx = reinterpret_cast<sdx_context_t*>(context);
  const jsize numInputs = env->GetArrayLength(inputData);
  const jsize numOutputs = env->GetArrayLength(outputShapes);

  // Copy inputs out of the JVM into stable host buffers.
  std::vector<std::vector<float>> inputs(numInputs);
  std::vector<std::vector<int64_t>> inShapes(numInputs);
  std::vector<sdx_tensor_view_t> inputViews(numInputs);
  for (jsize i = 0; i < numInputs; i++) {
    auto data = static_cast<jfloatArray>(env->GetObjectArrayElement(inputData, i));
    auto shape = static_cast<jlongArray>(env->GetObjectArrayElement(inputShapes, i));
    const jsize dataLen = env->GetArrayLength(data);
    const jsize rank = env->GetArrayLength(shape);
    inputs[i].resize(dataLen);
    inShapes[i].resize(rank);
    env->GetFloatArrayRegion(data, 0, dataLen, inputs[i].data());
    env->GetLongArrayRegion(shape, 0, rank, reinterpret_cast<jlong*>(inShapes[i].data()));
    env->DeleteLocalRef(data);
    env->DeleteLocalRef(shape);

    inputViews[i] = sdx_tensor_view_t{};
    inputViews[i].data = inputs[i].data();
    inputViews[i].shape = inShapes[i].data();
    inputViews[i].rank = rank;
    inputViews[i].dtype = 5;  // sd::DataType::FLOAT32
    inputViews[i].bytes = inputs[i].size() * sizeof(float);
    inputViews[i].device_type = SDX_DEVICE_HOST;
    inputViews[i].device_id = -1;
  }

  // Caller-provided output buffers from the declared specs.
  std::vector<std::vector<float>> outputs(numOutputs);
  std::vector<std::vector<int64_t>> outShapes(numOutputs);
  std::vector<sdx_tensor_view_t> outputViews(numOutputs);
  for (jsize i = 0; i < numOutputs; i++) {
    auto shape = static_cast<jlongArray>(env->GetObjectArrayElement(outputShapes, i));
    const jsize rank = env->GetArrayLength(shape);
    outShapes[i].resize(rank);
    env->GetLongArrayRegion(shape, 0, rank, reinterpret_cast<jlong*>(outShapes[i].data()));
    env->DeleteLocalRef(shape);

    int64_t elements = 1;
    for (int64_t d : outShapes[i]) elements *= d;
    outputs[i].assign(static_cast<size_t>(elements), 0.0f);

    outputViews[i] = sdx_tensor_view_t{};
    outputViews[i].data = outputs[i].data();
    outputViews[i].shape = outShapes[i].data();
    outputViews[i].rank = rank;
    outputViews[i].dtype = 5;
    outputViews[i].bytes = outputs[i].size() * sizeof(float);
    outputViews[i].device_type = SDX_DEVICE_HOST;
    outputViews[i].device_id = -1;
  }

  sdx_run_options_t options{};
  options.struct_size = sizeof(options);
  options.strict_signature = 1;
  const sdx_status_t status = sdxRun(
      ctx, inputViews.data(), static_cast<int32_t>(numInputs),
      outputViews.data(), static_cast<int32_t>(numOutputs), &options);
  if (status != SDX_STATUS_OK) {
    recordError("sdxRun", status);
    return nullptr;
  }

  jobjectArray result = env->NewObjectArray(
      numOutputs, env->FindClass("[F"), nullptr);
  for (jsize i = 0; i < numOutputs; i++) {
    jfloatArray arr = env->NewFloatArray(static_cast<jsize>(outputs[i].size()));
    env->SetFloatArrayRegion(arr, 0, static_cast<jsize>(outputs[i].size()), outputs[i].data());
    env->SetObjectArrayElement(result, i, arr);
    env->DeleteLocalRef(arr);
  }
  return result;
}

/** Report encoded as double[7]: status, requestedBackend, appliedBackend,
 *  usedFallback, executionTimeNs, planPhase, executionCount. */
JNIEXPORT jdoubleArray JNICALL
Java_org_nd4j_sdx_rn_SdxRuntimeModule_nativeGetExecutionReport(
    JNIEnv* env, jobject, jlong context) {
  sdx_execution_report_t report{};
  report.struct_size = sizeof(report);
  const sdx_status_t status =
      sdxGetExecutionReport(reinterpret_cast<sdx_context_t*>(context), &report);
  if (status != SDX_STATUS_OK) {
    recordError("sdxGetExecutionReport", status);
    return nullptr;
  }
  const double values[7] = {
      static_cast<double>(report.status_code),
      static_cast<double>(report.requested_backend),
      static_cast<double>(report.applied_backend),
      static_cast<double>(report.used_fallback),
      static_cast<double>(report.execution_time_ns),
      static_cast<double>(report.plan_phase),
      static_cast<double>(report.execution_count)};
  jdoubleArray result = env->NewDoubleArray(7);
  env->SetDoubleArrayRegion(result, 0, 7, values);
  return result;
}

}  // extern "C"
