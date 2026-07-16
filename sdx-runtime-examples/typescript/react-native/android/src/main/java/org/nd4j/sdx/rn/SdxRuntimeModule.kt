/*
 * ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 * ******************************************************************************
 */
package org.nd4j.sdx.rn

import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableArray
import java.io.File

/**
 * React Native bridge for the SDX runtime. All sdx* C ABI calls happen in the
 * JNI shim (src/main/cpp/sdx_jni.cpp); this class marshals React Native types
 * and manages the app-asset model path. Native handles cross the bridge as
 * integer ids (RN Promises carry doubles — safe for pointer values well below
 * 2^53, which is guaranteed on Android's 48-bit address space).
 */
class SdxRuntimeModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    companion object {
        init {
            System.loadLibrary("sdxrn")
        }
    }

    override fun getName(): String = "SdxRuntime"

    // ── JNI surface (implemented in sdx_jni.cpp) ────────────────────────────
    private external fun nativeAbiVersion(): Int
    private external fun nativeLastError(): String
    private external fun nativeLoadModel(path: String): Long
    private external fun nativeUnloadModel(model: Long)
    private external fun nativeCreateContext(model: Long, outputs: Array<String>): Long
    private external fun nativeDestroyContext(context: Long)
    private external fun nativeGetInputNames(context: Long): Array<String>
    private external fun nativeGetNumOutputs(context: Long): Int
    private external fun nativeMarkInputVariable(context: Long, index: Int): Int
    private external fun nativeMarkInputPlaceholder(context: Long, index: Int): Int
    private external fun nativeFreezeShapes(context: Long): Int
    private external fun nativeGetPlanPhase(context: Long): Int
    private external fun nativeGetExecutionCount(context: Long): Int
    private external fun nativeRun(
        context: Long,
        inputData: Array<FloatArray>,
        inputShapes: Array<LongArray>,
        outputShapes: Array<LongArray>
    ): Array<FloatArray>?
    private external fun nativeGetExecutionReport(context: Long): DoubleArray?

    // ── React methods (mirror src/index.ts) ─────────────────────────────────

    @ReactMethod
    fun getAbiVersion(promise: Promise) {
        promise.resolve(nativeAbiVersion())
    }

    /** Copies an app asset (e.g. mlp.sdz) into the files dir and returns its path. */
    @ReactMethod
    fun resolveModelAsset(assetName: String, promise: Promise) {
        try {
            val target = File(reactApplicationContext.filesDir, assetName)
            reactApplicationContext.assets.open(assetName).use { input ->
                target.outputStream().use { output -> input.copyTo(output) }
            }
            promise.resolve(target.absolutePath)
        } catch (e: Exception) {
            promise.reject("E_ASSET", "Cannot resolve asset '$assetName': ${e.message}", e)
        }
    }

    @ReactMethod
    fun loadModel(path: String, promise: Promise) {
        val handle = nativeLoadModel(path)
        if (handle == 0L) {
            promise.reject("E_LOAD", nativeLastError())
        } else {
            promise.resolve(handle.toDouble())
        }
    }

    @ReactMethod
    fun unloadModel(modelId: Double, promise: Promise) {
        nativeUnloadModel(modelId.toLong())
        promise.resolve(null)
    }

    @ReactMethod
    fun createContext(modelId: Double, requestedOutputs: ReadableArray, promise: Promise) {
        val outputs = Array(requestedOutputs.size()) { requestedOutputs.getString(it) ?: "" }
        val handle = nativeCreateContext(modelId.toLong(), outputs)
        if (handle == 0L) {
            promise.reject("E_CONTEXT", nativeLastError())
        } else {
            promise.resolve(handle.toDouble())
        }
    }

    @ReactMethod
    fun destroyContext(contextId: Double, promise: Promise) {
        nativeDestroyContext(contextId.toLong())
        promise.resolve(null)
    }

    @ReactMethod
    fun getInputNames(contextId: Double, promise: Promise) {
        val result = Arguments.createArray()
        nativeGetInputNames(contextId.toLong()).forEach { result.pushString(it) }
        promise.resolve(result)
    }

    @ReactMethod
    fun getNumOutputs(contextId: Double, promise: Promise) {
        promise.resolve(nativeGetNumOutputs(contextId.toLong()))
    }

    @ReactMethod
    fun markInputVariable(contextId: Double, index: Double, promise: Promise) {
        val status = nativeMarkInputVariable(contextId.toLong(), index.toInt())
        if (status != 0) promise.reject("E_MARK", nativeLastError()) else promise.resolve(null)
    }

    @ReactMethod
    fun markInputPlaceholder(contextId: Double, index: Double, promise: Promise) {
        val status = nativeMarkInputPlaceholder(contextId.toLong(), index.toInt())
        if (status != 0) promise.reject("E_MARK", nativeLastError()) else promise.resolve(null)
    }

    @ReactMethod
    fun run(contextId: Double, inputs: ReadableArray, outputShapes: ReadableArray, promise: Promise) {
        try {
            val n = inputs.size()
            val inputData = Array(n) { i ->
                val tensor = inputs.getMap(i)!!
                val data = tensor.getArray("data")!!
                FloatArray(data.size()) { j -> data.getDouble(j).toFloat() }
            }
            val inputShapeArr = Array(n) { i ->
                val tensor = inputs.getMap(i)!!
                val shape = tensor.getArray("shape")!!
                LongArray(shape.size()) { j -> shape.getDouble(j).toLong() }
            }
            val outShapeArr = Array(outputShapes.size()) { i ->
                val shape = outputShapes.getArray(i)!!
                LongArray(shape.size()) { j -> shape.getDouble(j).toLong() }
            }

            val outputs = nativeRun(contextId.toLong(), inputData, inputShapeArr, outShapeArr)
                ?: return promise.reject("E_RUN", nativeLastError())

            val result = Arguments.createArray()
            outputs.forEach { floats ->
                val arr = Arguments.createArray()
                floats.forEach { arr.pushDouble(it.toDouble()) }
                result.pushArray(arr)
            }
            promise.resolve(result)
        } catch (e: Exception) {
            promise.reject("E_RUN", e.message, e)
        }
    }

    @ReactMethod
    fun freezeShapes(contextId: Double, promise: Promise) {
        val status = nativeFreezeShapes(contextId.toLong())
        if (status != 0) promise.reject("E_FREEZE", nativeLastError()) else promise.resolve(null)
    }

    @ReactMethod
    fun getPlanPhase(contextId: Double, promise: Promise) {
        promise.resolve(nativeGetPlanPhase(contextId.toLong()))
    }

    @ReactMethod
    fun getExecutionCount(contextId: Double, promise: Promise) {
        promise.resolve(nativeGetExecutionCount(contextId.toLong()))
    }

    @ReactMethod
    fun getExecutionReport(contextId: Double, promise: Promise) {
        val values = nativeGetExecutionReport(contextId.toLong())
            ?: return promise.reject("E_REPORT", nativeLastError())
        val report = Arguments.createMap()
        report.putInt("statusCode", values[0].toInt())
        report.putInt("requestedBackend", values[1].toInt())
        report.putInt("appliedBackend", values[2].toInt())
        report.putInt("usedFallback", values[3].toInt())
        report.putDouble("executionTimeNs", values[4])
        report.putInt("planPhase", values[5].toInt())
        report.putInt("executionCount", values[6].toInt())
        promise.resolve(report)
    }
}
