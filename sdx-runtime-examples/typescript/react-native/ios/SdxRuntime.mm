/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

// React Native bridge for the SDX runtime on iOS. ObjC++ calls the sdx* C ABI
// directly — link the SDK's ND4JDSPRuntime .xcframework (or the runtime dylib)
// into the app target and add the SDK include/ dir to HEADER_SEARCH_PATHS so
// <dsp/runtime/dsp_runtime_c.h> resolves.
//
// Handles cross the bridge as NSNumber doubles (safe below 2^53).

#import <React/RCTBridgeModule.h>

#include <dsp/runtime/dsp_runtime_c.h>

#include <mutex>
#include <string>
#include <vector>

static sdx_runtime_t *gRuntime = nullptr;
static std::mutex gRuntimeMutex;
static std::string gLastError;

static sdx_runtime_t *SdxSharedRuntime(void) {
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

static void SdxRecordError(const char *op, sdx_status_t status) {
  const char *detail = gRuntime != nullptr ? sdxGetLastError(gRuntime) : "";
  gLastError = std::string(op) + " failed: status=" + std::to_string(status) +
               (detail != nullptr && detail[0] != '\0' ? std::string(", error=") + detail : "");
}

@interface SdxRuntime : NSObject <RCTBridgeModule>
@end

@implementation SdxRuntime

RCT_EXPORT_MODULE(SdxRuntime)

+ (BOOL)requiresMainQueueSetup {
  return NO;
}

RCT_EXPORT_METHOD(getAbiVersion:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  resolve(@(sdxGetRuntimeAbiVersion()));
}

RCT_EXPORT_METHOD(resolveModelAsset:(NSString *)assetName
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  NSString *base = [assetName stringByDeletingPathExtension];
  NSString *ext = [assetName pathExtension];
  NSString *path = [[NSBundle mainBundle] pathForResource:base ofType:ext];
  if (path == nil) {
    reject(@"E_ASSET",
           [NSString stringWithFormat:@"Bundle resource '%@' not found", assetName], nil);
    return;
  }
  resolve(path);
}

RCT_EXPORT_METHOD(loadModel:(NSString *)path
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  sdx_runtime_t *runtime = SdxSharedRuntime();
  if (runtime == nullptr) {
    reject(@"E_LOAD", @"sdxCreateRuntime failed", nil);
    return;
  }
  sdx_model_options_t options{};
  options.struct_size = sizeof(options);
  sdx_model_t *model = nullptr;
  sdx_status_t status = sdxLoadBundle(runtime, path.UTF8String, &options, &model);
  if (status != SDX_STATUS_OK) {
    SdxRecordError("sdxLoadBundle", status);
    reject(@"E_LOAD", @(gLastError.c_str()), nil);
    return;
  }
  resolve(@((double)(uintptr_t)model));
}

RCT_EXPORT_METHOD(unloadModel:(nonnull NSNumber *)modelId
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  sdxUnloadModel((sdx_model_t *)(uintptr_t)modelId.doubleValue);
  resolve(nil);
}

RCT_EXPORT_METHOD(createContext:(nonnull NSNumber *)modelId
                  requestedOutputs:(NSArray<NSString *> *)requestedOutputs
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  std::vector<std::string> names;
  std::vector<const char *> namePtrs;
  for (NSString *name in requestedOutputs) {
    names.emplace_back(name.UTF8String);
  }
  for (auto &name : names) {
    namePtrs.push_back(name.c_str());
  }
  sdx_context_t *context = nullptr;
  sdx_status_t status = sdxCreateContext(
      (sdx_model_t *)(uintptr_t)modelId.doubleValue,
      namePtrs.empty() ? nullptr : namePtrs.data(),
      (int32_t)namePtrs.size(), &context);
  if (status != SDX_STATUS_OK) {
    SdxRecordError("sdxCreateContext", status);
    reject(@"E_CONTEXT", @(gLastError.c_str()), nil);
    return;
  }
  resolve(@((double)(uintptr_t)context));
}

RCT_EXPORT_METHOD(destroyContext:(nonnull NSNumber *)contextId
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  sdxDestroyContext((sdx_context_t *)(uintptr_t)contextId.doubleValue);
  resolve(nil);
}

RCT_EXPORT_METHOD(getInputNames:(nonnull NSNumber *)contextId
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  sdx_context_t *ctx = (sdx_context_t *)(uintptr_t)contextId.doubleValue;
  int32_t numInputs = sdxGetNumInputs(ctx);
  NSMutableArray<NSString *> *names = [NSMutableArray arrayWithCapacity:MAX(numInputs, 0)];
  for (int32_t i = 0; i < numInputs; i++) {
    const char *name = sdxGetInputName(ctx, i);
    [names addObject:name != nullptr ? @(name) : @""];
  }
  resolve(names);
}

RCT_EXPORT_METHOD(getNumOutputs:(nonnull NSNumber *)contextId
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  resolve(@(sdxGetNumOutputs((sdx_context_t *)(uintptr_t)contextId.doubleValue)));
}

RCT_EXPORT_METHOD(markInputVariable:(nonnull NSNumber *)contextId
                  index:(nonnull NSNumber *)index
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  sdx_status_t status = sdxMarkInputVariable(
      (sdx_context_t *)(uintptr_t)contextId.doubleValue, index.intValue);
  if (status != SDX_STATUS_OK) {
    SdxRecordError("sdxMarkInputVariable", status);
    reject(@"E_MARK", @(gLastError.c_str()), nil);
    return;
  }
  resolve(nil);
}

RCT_EXPORT_METHOD(markInputPlaceholder:(nonnull NSNumber *)contextId
                  index:(nonnull NSNumber *)index
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  sdx_status_t status = sdxMarkInputPlaceholder(
      (sdx_context_t *)(uintptr_t)contextId.doubleValue, index.intValue);
  if (status != SDX_STATUS_OK) {
    SdxRecordError("sdxMarkInputPlaceholder", status);
    reject(@"E_MARK", @(gLastError.c_str()), nil);
    return;
  }
  resolve(nil);
}

RCT_EXPORT_METHOD(run:(nonnull NSNumber *)contextId
                  inputs:(NSArray *)inputs
                  outputShapes:(NSArray *)outputShapes
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  sdx_context_t *ctx = (sdx_context_t *)(uintptr_t)contextId.doubleValue;
  const NSUInteger numInputs = inputs.count;
  const NSUInteger numOutputs = outputShapes.count;

  std::vector<std::vector<float>> inputData(numInputs);
  std::vector<std::vector<int64_t>> inputShapes(numInputs);
  std::vector<sdx_tensor_view_t> inputViews(numInputs);
  for (NSUInteger i = 0; i < numInputs; i++) {
    NSDictionary *tensor = inputs[i];
    NSArray *data = tensor[@"data"];
    NSArray *shape = tensor[@"shape"];
    inputData[i].reserve(data.count);
    inputShapes[i].reserve(shape.count);
    for (NSNumber *v in data) inputData[i].push_back(v.floatValue);
    for (NSNumber *d in shape) inputShapes[i].push_back(d.longLongValue);

    inputViews[i] = sdx_tensor_view_t{};
    inputViews[i].data = inputData[i].data();
    inputViews[i].shape = inputShapes[i].data();
    inputViews[i].rank = (int32_t)inputShapes[i].size();
    inputViews[i].dtype = 5;  // sd::DataType::FLOAT32
    inputViews[i].bytes = inputData[i].size() * sizeof(float);
    inputViews[i].device_type = SDX_DEVICE_HOST;
    inputViews[i].device_id = -1;
  }

  std::vector<std::vector<float>> outputData(numOutputs);
  std::vector<std::vector<int64_t>> outShapes(numOutputs);
  std::vector<sdx_tensor_view_t> outputViews(numOutputs);
  for (NSUInteger i = 0; i < numOutputs; i++) {
    NSArray *shape = outputShapes[i];
    int64_t elements = 1;
    for (NSNumber *d in shape) {
      outShapes[i].push_back(d.longLongValue);
      elements *= d.longLongValue;
    }
    outputData[i].assign((size_t)elements, 0.0f);

    outputViews[i] = sdx_tensor_view_t{};
    outputViews[i].data = outputData[i].data();
    outputViews[i].shape = outShapes[i].data();
    outputViews[i].rank = (int32_t)outShapes[i].size();
    outputViews[i].dtype = 5;
    outputViews[i].bytes = outputData[i].size() * sizeof(float);
    outputViews[i].device_type = SDX_DEVICE_HOST;
    outputViews[i].device_id = -1;
  }

  sdx_run_options_t options{};
  options.struct_size = sizeof(options);
  options.strict_signature = 1;
  sdx_status_t status = sdxRun(ctx, inputViews.data(), (int32_t)numInputs,
                               outputViews.data(), (int32_t)numOutputs, &options);
  if (status != SDX_STATUS_OK) {
    SdxRecordError("sdxRun", status);
    reject(@"E_RUN", @(gLastError.c_str()), nil);
    return;
  }

  NSMutableArray *result = [NSMutableArray arrayWithCapacity:numOutputs];
  for (NSUInteger i = 0; i < numOutputs; i++) {
    NSMutableArray *values = [NSMutableArray arrayWithCapacity:outputData[i].size()];
    for (float v : outputData[i]) [values addObject:@(v)];
    [result addObject:values];
  }
  resolve(result);
}

RCT_EXPORT_METHOD(freezeShapes:(nonnull NSNumber *)contextId
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  sdx_status_t status = sdxFreezeShapes((sdx_context_t *)(uintptr_t)contextId.doubleValue);
  if (status != SDX_STATUS_OK) {
    SdxRecordError("sdxFreezeShapes", status);
    reject(@"E_FREEZE", @(gLastError.c_str()), nil);
    return;
  }
  resolve(nil);
}

RCT_EXPORT_METHOD(getPlanPhase:(nonnull NSNumber *)contextId
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  resolve(@(sdxGetPlanPhase((sdx_context_t *)(uintptr_t)contextId.doubleValue)));
}

RCT_EXPORT_METHOD(getExecutionCount:(nonnull NSNumber *)contextId
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  resolve(@(sdxGetExecutionCount((sdx_context_t *)(uintptr_t)contextId.doubleValue)));
}

RCT_EXPORT_METHOD(getExecutionReport:(nonnull NSNumber *)contextId
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  sdx_execution_report_t report{};
  report.struct_size = sizeof(report);
  sdx_status_t status = sdxGetExecutionReport(
      (sdx_context_t *)(uintptr_t)contextId.doubleValue, &report);
  if (status != SDX_STATUS_OK) {
    SdxRecordError("sdxGetExecutionReport", status);
    reject(@"E_REPORT", @(gLastError.c_str()), nil);
    return;
  }
  resolve(@{
    @"statusCode" : @(report.status_code),
    @"requestedBackend" : @(report.requested_backend),
    @"appliedBackend" : @(report.applied_backend),
    @"usedFallback" : @(report.used_fallback),
    @"executionTimeNs" : @((double)report.execution_time_ns),
    @"planPhase" : @(report.plan_phase),
    @"executionCount" : @(report.execution_count),
  });
}

@end
