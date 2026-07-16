/*******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.examples.advanced.devicemanagement;

import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceMemoryManager.DeviceRoutingPolicy;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.factory.Nd4j;

/**
 * GPU device failover and multi-device memory management with {@link DeviceMemoryManager}.
 *
 * ND4J routes allocations across devices and, when an allocation cannot be satisfied on
 * the requested GPU, fails over to another device instead of crashing with an OOM:
 *
 *   1. The CUDA allocator asks {@link DeviceMemoryManager#selectFailoverDevice(long, int)}
 *      for the best OTHER GPU (ranked by pool-aware free memory).
 *   2. If no GPU fits, it falls back to the CPU (when auto-fallback is enabled).
 *   3. Only when nothing fits is the real OOM surfaced to the caller.
 *
 * On real CUDA systems this happens automatically inside the native allocator — including
 * on multi-GPU boxes WITHOUT peer access (e.g. mixed consumer GPUs), where cross-device
 * memory is served via CUDA unified memory (cudaMallocManaged) so non-peer devices are
 * still valid failover targets.
 *
 * This example uses the manager's memory-simulation mode so the failover logic can be
 * demonstrated and tested on any machine (CPU-only included). The same calls drive the
 * real routing decisions on CUDA. Related production knobs:
 *
 *   - Nd4j.getEnvironment().setMaxDeviceMemory(bytes): hard per-device budget enforced by
 *     the CUDA memory pool — allocations beyond the budget fail over instead of growing.
 *   - DeviceMemoryManager.configureStubTopology(...): simulate multi-GPU topologies in
 *     tests on single-GPU machines (see DspReplayDeviceAnalyticsTest in platform-tests).
 *
 * Run: mvn exec:java -Dexec.mainClass=org.nd4j.examples.advanced.devicemanagement.GpuDeviceFailoverExample
 */
public class GpuDeviceFailoverExample {

    private static final long GB = 1024L * 1024L * 1024L;
    private static final long MB = 1024L * 1024L;

    public static void main(String[] args) {
        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
        mgr.clearDevices();

        try {
            // ============================================================
            // 1. REGISTER A DEVICE TOPOLOGY
            // ============================================================
            System.out.println("=== 1. Device registration and memory caps ===");

            DeviceDescriptor cpu = DeviceDescriptor.cpu();
            DeviceDescriptor gpu0 = DeviceDescriptor.cuda(0);   // e.g. a 24GB card
            DeviceDescriptor gpu1 = DeviceDescriptor.cuda(1);   // e.g. an 8GB card

            mgr.registerDevice(cpu);
            mgr.registerDevice(gpu0);
            mgr.registerDevice(gpu1);

            // Caps bound how much the manager will place on each device
            mgr.setMemoryCap(cpu, 32 * GB);
            mgr.setMemoryCap(gpu0, 24 * GB);
            mgr.setMemoryCap(gpu1, 8 * GB);

            // Priorities break ties: higher = preferred
            mgr.setDevicePriority(gpu0, 10);
            mgr.setDevicePriority(gpu1, 5);
            mgr.setDefaultDevice(gpu0);
            mgr.setFallbackDevice(cpu);

            System.out.println("  Registered devices: " + mgr.getRegisteredDeviceCount());
            for (DeviceDescriptor d : mgr.getRegisteredDevices()) {
                System.out.println("    " + d.getDeviceId()
                        + "  cap=" + (mgr.getMemoryCap(d) / GB) + "GB"
                        + "  priority=" + mgr.getDevicePriority(d));
            }
            check(mgr.getRegisteredDeviceCount() == 3, "expected 3 registered devices");

            // ============================================================
            // 2. ROUTING POLICIES
            // ============================================================
            System.out.println("\n=== 2. Routing policies ===");

            DeviceDescriptor preferGpu = mgr.selectDevice(1 * GB, DeviceRoutingPolicy.PREFER_GPU);
            DeviceDescriptor preferCpu = mgr.selectDevice(1 * GB, DeviceRoutingPolicy.PREFER_CPU);
            DeviceDescriptor byPriority = mgr.selectDevice(1 * GB, DeviceRoutingPolicy.MEMORY_PRIORITY);

            System.out.println("  PREFER_GPU      -> " + preferGpu.getDeviceId());
            System.out.println("  PREFER_CPU      -> " + preferCpu.getDeviceId());
            System.out.println("  MEMORY_PRIORITY -> " + byPriority.getDeviceId());

            check(preferGpu.getDeviceType() == DeviceType.CUDA_GPU, "PREFER_GPU must pick a GPU");
            check(preferCpu.getDeviceType() == DeviceType.CPU, "PREFER_CPU must pick the CPU");

            // ============================================================
            // 3. ALLOCATION TRACKING, UTILIZATION AND canAllocate
            // ============================================================
            System.out.println("\n=== 3. Allocation tracking ===");

            mgr.recordAllocation(gpu1, 6 * GB);                    // 6GB of the 8GB cap in use
            double util = mgr.getMemoryUtilization(gpu1);
            System.out.println("  gpu1 allocated: " + (mgr.getAllocatedMemory(gpu1) / GB) + "GB"
                    + "  utilization: " + String.format("%.0f%%", util * 100));
            System.out.println("  canAllocate(gpu1, 1GB) = " + mgr.canAllocate(gpu1, 1 * GB));
            System.out.println("  canAllocate(gpu1, 4GB) = " + mgr.canAllocate(gpu1, 4 * GB));

            check(mgr.canAllocate(gpu1, 1 * GB), "1GB should fit in the remaining 2GB");
            check(!mgr.canAllocate(gpu1, 4 * GB), "4GB must NOT fit in the remaining 2GB");

            // ============================================================
            // 4. MEMORY PRESSURE CALLBACKS
            // ============================================================
            System.out.println("\n=== 4. Memory pressure callbacks ===");

            final boolean[] pressureFired = {false};
            DeviceMemoryManager.MemoryPressureCallback callback = (device, utilization) -> {
                pressureFired[0] = true;
                System.out.println("  [callback] pressure on " + device.getDeviceId()
                        + " at " + String.format("%.0f%%", utilization * 100)
                        + " — a serving system would shed load or migrate here");
            };
            mgr.setMemoryPressureThreshold(0.9);
            mgr.addMemoryPressureCallback(callback);

            mgr.recordAllocation(gpu1, 1536 * MB);                 // push gpu1 past 90%
            check(pressureFired[0], "pressure callback must fire above the 90% threshold");
            mgr.removeMemoryPressureCallback(callback);
            mgr.recordDeallocation(gpu1, 6 * GB + 1536 * MB);      // release for the next section

            // ============================================================
            // 5. OOM FAILOVER — the core scenario
            // ============================================================
            System.out.println("\n=== 5. OOM failover ===");

            // Simulation mode drives the same selection logic the CUDA allocator invokes
            // on a real allocation failure — reproducible on any machine.
            mgr.setSimulatedFreeMemory(0, 10 * MB);                          // GPU 0: nearly full
            mgr.setSimulatedFreeMemory(1, 6 * GB);                           // GPU 1: lots of room
            mgr.setSimulatedFreeMemory(DeviceMemoryManager.CPU_DEVICE_ID, 32 * GB);
            mgr.setMemorySimulationEnabled(true);

            try {
                // (a) GPU 0 cannot fit 200MB -> the failover target must be GPU 1
                DeviceDescriptor target = mgr.selectFailoverDevice(200 * MB, 0);
                System.out.println("  200MB, GPU0 full          -> " + target.getDeviceId());
                check(target.getDeviceType() == DeviceType.CUDA_GPU
                        && target.getDeviceIndex() == 1, "failover should pick GPU 1");

                // (b) All GPUs full -> CPU fallback keeps the job alive
                mgr.setSimulatedFreeMemory(1, 10 * MB);
                target = mgr.selectFailoverDevice(200 * MB, 0);
                System.out.println("  200MB, all GPUs full      -> " + target.getDeviceId());
                check(target.getDeviceType() == DeviceType.CPU, "failover should reach the CPU");

                // (c) Nothing fits anywhere -> null, so the caller surfaces the real OOM
                mgr.setSimulatedFreeMemory(DeviceMemoryManager.CPU_DEVICE_ID, 10 * MB);
                target = mgr.selectFailoverDevice(200 * MB, 0);
                System.out.println("  200MB, nothing fits       -> " + target);
                check(target == null, "when nothing fits, failover must return null");
            } finally {
                mgr.clearAllMemorySimulation();
                mgr.setMemorySimulationEnabled(false);
            }

            // ============================================================
            // 6. REAL-BACKEND INTEGRATION
            // ============================================================
            System.out.println("\n=== 6. Real-backend integration ===");

            if (Nd4j.getEnvironment().isCPU()) {
                System.out.println("  Running on the CPU backend — on nd4j-cuda the pieces above engage");
                System.out.println("  automatically: the allocator calls selectFailoverDevice(...) on OOM,");
                System.out.println("  non-peer GPUs are reached via unified memory, and per-device budgets");
                System.out.println("  set with Nd4j.getEnvironment().setMaxDeviceMemory(bytes) trigger");
                System.out.println("  failover before a device is exhausted.");
            } else {
                // On a CUDA backend, inspect the live view the failover logic uses.
                int devices = Nd4j.getAffinityManager().getNumberOfDevices();
                System.out.println("  CUDA backend with " + devices + " device(s)");
                for (int i = 0; i < devices; i++) {
                    System.out.println("    GPU " + i + " pool-aware free: "
                            + (mgr.getPoolAwareFreeMemory(i) / MB) + "MB");
                }
                // A per-device budget: allocations beyond this fail over instead of growing.
                // Uncomment to enforce e.g. a 4GB budget on the current device:
                // Nd4j.getEnvironment().setMaxDeviceMemory(4 * GB);
            }

            System.out.println("\nAll device failover scenarios verified successfully.");
        } finally {
            mgr.clearDevices();
        }
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException("FAILED: " + message);
        }
    }
}
