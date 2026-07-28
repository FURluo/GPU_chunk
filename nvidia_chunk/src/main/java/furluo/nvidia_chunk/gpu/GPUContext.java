package furluo.nvidia_chunk.gpu;

import com.mojang.logging.LogUtils;
import furluo.nvidia_chunk.Config;
import org.jocl.CL;
import org.jocl.CLException;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;
import org.slf4j.Logger;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.ConcurrentModificationException;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;

import static org.jocl.CL.*;

/**
 * OpenCL GPU 上下文管理（单例）。
 *
 * <p>负责：</p>
 * <ul>
 *   <li>枚举 OpenCL 平台与设备，按 {@link Config#platformPreference} 选择 NVIDIA / AMD / Intel</li>
 *   <li>编译 ImprovedNoise 内核（与原版 {@code net.minecraft.world.level.levelgen.synth.ImprovedNoise}
 *       逐字节码对应，确保输出 100% 一致）</li>
 *   <li>为每个区块工作线程维护独立的命令队列与可重用缓冲区</li>
 *   <li>提供 {@link #dispatchBatch} 批量调度与 {@link #dispatchSingle} 单次调度接口</li>
 * </ul>
 *
 * <p>线程模型：{@code cl_context} / {@code cl_program} / {@code cl_kernel} 可被多线程共享；
 * {@code cl_command_queue} 与 {@code cl_mem} 缓冲区为每线程独有（通过 {@link ThreadLocal}）。</p>
 *
 * <p>失败处理：任何 OpenCL 错误都会将 {@link #available} 置为 false，
 * 上层 {@link GPUNoiseManager} 会自动回退到 CPU 路径。</p>
 */
public class GPUContext {

    private static final Logger LOGGER = LogUtils.getLogger();

    /**
     * ImprovedNoise OpenCL 内核源码。
     *
     * <p>严格按 MC 1.20.1 {@code ImprovedNoise.noise(DDDDD)D} 与 {@code sampleAndLerp} 字节码移植：</p>
     * <ul>
     *   <li>yOffset 公式：{@code floor(d7 / yFloorFreq + 1.0000000116860974E-7) * yFloorFreq}，
     *       其中 {@code d7 = (yFloorValue >= 0 && yFloorValue < fracY) ? yFloorValue : fracY}</li>
     *   <li>梯度点积使用 {@code fracY - yOffset}（修改后），smoothstep 使用原始 {@code fracY}</li>
     *   <li>GRADIENT 表与 {@code SimplexNoise.GRADIENT} 完全一致</li>
     *   <li>{@code p(i) = perm[i & 255] & 255}（uchar 已经无符号，& 255 仅保险）</li>
     *   <li>{@code floor} 与 Java {@code Mth.floor} 一致：{@code (int)v; return v < i ? i-1 : i}</li>
     *   <li>{@code smoothstep(t) = t*t*t * (t*(t*6-15)+10)}</li>
     *   <li>{@code lerp(delta,a,b) = a + delta*(b-a)}（不是 {@code a*(1-d)+b*d}，IEEE 浮点结果不同）</li>
     * </ul>
     */
    static final String KERNEL_SOURCE = """
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable

        __constant int GRADIENT[16][3] = {
            { 1,  1,  0}, {-1,  1,  0}, { 1, -1,  0}, {-1, -1,  0},
            { 1,  0,  1}, {-1,  0,  1}, { 1,  0, -1}, {-1,  0, -1},
            { 0,  1,  1}, { 0, -1,  1}, { 0,  1, -1}, { 0, -1, -1},
            { 1,  1,  0}, { 0, -1,  1}, {-1,  1,  0}, { 0, -1, -1}
        };

        // 等价于 Java Mth.floor(double)：向零截断后，负数再 -1
        int mc_floor(double v) {
            int i = (int)v;
            return (v < (double)i) ? i - 1 : i;
        }

        // 等价于 ImprovedNoise.p(int)：perm[i & 255] & 255
        int p_idx(__global const uchar* perm, int i) {
            return (int)(perm[i & 255]);
        }

        // 等价于 SimplexNoise.dot(int[], double, double, double) + ImprovedNoise.gradDot
        double grad_dot(int hash, double x, double y, double z) {
            int idx = hash & 15;
            return (double)GRADIENT[idx][0] * x
                 + (double)GRADIENT[idx][1] * y
                 + (double)GRADIENT[idx][2] * z;
        }

        // 等价于 Mth.smoothstep：6t^5 - 15t^4 + 10t^3
        double smoothstep(double t) {
            return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        }

        // 等价于 Mth.lerp(delta, a, b) = a + delta * (b - a)
        double lerp(double delta, double a, double b) {
            return a + delta * (b - a);
        }

        // 等价于 Mth.lerp2
        double lerp2(double p1, double p2, double p3, double p4, double p5, double p6) {
            return lerp(p2, lerp(p1, p3, p4), lerp(p1, p5, p6));
        }

        // 等价于 Mth.lerp3
        double lerp3(double p1, double p2, double p3,
                     double p4, double p5, double p6, double p7,
                     double p8, double p9, double p10, double p11) {
            return lerp(p3,
                        lerp2(p1, p2, p4, p5, p6, p7),
                        lerp2(p1, p2, p8, p9, p10, p11));
        }

        // 主内核：每个 work-item 计算一次 ImprovedNoise.noise(x, y, z, yFloorFreq, yFloorValue)
        __kernel void improved_noise_kernel(
            __global const uchar* permutation,   // 256 字节置换表
            const double xo,
            const double yo,
            const double zo,
            __global const double* inputs,        // count * 5 doubles
            __global double* outputs,             // count doubles
            const int count
        ) {
            int idx = get_global_id(0);
            if (idx >= count) return;

            double x            = inputs[idx * 5 + 0];
            double y            = inputs[idx * 5 + 1];
            double z            = inputs[idx * 5 + 2];
            double yFloorFreq   = inputs[idx * 5 + 3];
            double yFloorValue  = inputs[idx * 5 + 4];

            double x2 = x + xo;
            double y2 = y + yo;
            double z2 = z + zo;

            int floorX = mc_floor(x2);
            int floorY = mc_floor(y2);
            int floorZ = mc_floor(z2);

            double fracX = x2 - (double)floorX;
            double fracY = y2 - (double)floorY;
            double fracZ = z2 - (double)floorZ;

            double yOffset;
            if (yFloorFreq != 0.0) {
                double d7 = (yFloorValue >= 0.0 && yFloorValue < fracY) ? yFloorValue : fracY;
                yOffset = (double)mc_floor(d7 / yFloorFreq + 1.0000000116860974E-7) * yFloorFreq;
            } else {
                yOffset = 0.0;
            }

            double fracYmod  = fracY - yOffset;
            double fracYorig = fracY;

            int pi  = p_idx(permutation, floorX);
            int pj  = p_idx(permutation, floorX + 1);
            int pk  = p_idx(permutation, pi + floorY);
            int pl  = p_idx(permutation, pi + floorY + 1);
            int pi1 = p_idx(permutation, pj + floorY);
            int pj1 = p_idx(permutation, pj + floorY + 1);

            double d0 = grad_dot(p_idx(permutation, pk  + floorZ),     fracX,     fracYmod,     fracZ    );
            double d1 = grad_dot(p_idx(permutation, pi1 + floorZ),     fracX - 1, fracYmod,     fracZ    );
            double d2 = grad_dot(p_idx(permutation, pl  + floorZ),     fracX,     fracYmod - 1, fracZ    );
            double d3 = grad_dot(p_idx(permutation, pj1 + floorZ),     fracX - 1, fracYmod - 1, fracZ    );
            double d4 = grad_dot(p_idx(permutation, pk  + floorZ + 1), fracX,     fracYmod,     fracZ - 1);
            double d5 = grad_dot(p_idx(permutation, pi1 + floorZ + 1), fracX - 1, fracYmod,     fracZ - 1);
            double d6 = grad_dot(p_idx(permutation, pl  + floorZ + 1), fracX,     fracYmod - 1, fracZ - 1);
            double d7 = grad_dot(p_idx(permutation, pj1 + floorZ + 1), fracX - 1, fracYmod - 1, fracZ - 1);

            double sx = smoothstep(fracX);
            double sy = smoothstep(fracYorig);
            double sz = smoothstep(fracZ);

            outputs[idx] = lerp3(sx, sy, sz, d0, d1, d2, d3, d4, d5, d6, d7);
        }
        """;

    // ------------------------------------------------------------------
    // 单例
    // ------------------------------------------------------------------
    private static volatile GPUContext instance;

    // ------------------------------------------------------------------
    // OpenCL 句柄（共享，线程安全）
    // ------------------------------------------------------------------
    private cl_context context;
    private cl_device_id device;
    private cl_program program;
    private cl_kernel kernel;
    private final int workGroupSize;
    private final String vendorName;
    private final String deviceName;
    private volatile boolean available;
    private volatile boolean closed;

    // ------------------------------------------------------------------
    // 每线程状态：命令队列 + 可重用缓冲区
    // ------------------------------------------------------------------
    private final ThreadLocal<PerThreadState> perThread = new ThreadLocal<>();

    private final ReferenceQueue<PerThreadState> cleanupQueue = new ReferenceQueue<>();
    private final Set<StateReference> trackedStates = Collections.newSetFromMap(new ConcurrentHashMap<>());
    private volatile Thread cleanupThread;

    private static class StateReference extends PhantomReference<PerThreadState> {
        final PerThreadState state;
        StateReference(PerThreadState state, ReferenceQueue<PerThreadState> queue) {
            super(state, queue);
            this.state = state;
        }
    }

    private void startCleanupThread() {
        if (cleanupThread != null) return;
        cleanupThread = new Thread(() -> {
            while (!Thread.currentThread().isInterrupted()) {
                try {
                    StateReference ref = (StateReference) cleanupQueue.remove(5000);
                    if (ref != null) {
                        releasePerThreadState(ref.state);
                        trackedStates.remove(ref);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                } catch (Throwable t) {
                    LOGGER.warn("[nvidia_chunk] 资源清理线程异常：{}", t.getMessage());
                }
            }
        }, "nvidia_chunk-cleanup");
        cleanupThread.setDaemon(true);
        cleanupThread.start();
    }

    // ------------------------------------------------------------------
    // 厂商匹配关键词（按 CL_VENDOR 字符串匹配）
    // ------------------------------------------------------------------
    private static final String[][] VENDOR_KEYWORDS = {
            {"nvidia", "NVIDIA Corporation", "GeForce", "Quadro", "Tesla"},
            {"amd", "Advanced Micro Devices", "AMD", "Radeon", "gfx"},
            {"intel", "Intel Corporation", "Intel", "Arc", "Iris", "HD Graphics"}
    };

    private GPUContext(cl_context context, cl_device_id device, cl_program program,
                       cl_kernel kernel, int workGroupSize, String vendorName, String deviceName) {
        this.context = context;
        this.device = device;
        this.program = program;
        this.kernel = kernel;
        this.workGroupSize = workGroupSize;
        this.vendorName = vendorName;
        this.deviceName = deviceName;
        this.available = true;
    }

    /**
     * 初始化 GPU 上下文。线程安全，仅应调用一次。
     *
     * @param preferredVendor "auto" / "nvidia" / "amd" / "intel"
     * @param deviceIndex     在选定厂商平台上的设备序号
     * @param workGroupSize   期望的工作组大小（实际取 min(期望, 设备上限)）
     * @return 初始化后的实例；失败返回 null
     */
    public static synchronized GPUContext init(String preferredVendor, int deviceIndex, int workGroupSize) {
        if (instance != null) {
            LOGGER.warn("[nvidia_chunk] GPUContext 已初始化，忽略重复调用");
            return instance;
        }
        try {
            CL.setExceptionsEnabled(true);

            int[] numPlatformsArray = new int[1];
            clGetPlatformIDs(0, null, numPlatformsArray);
            int numPlatforms = numPlatformsArray[0];
            if (numPlatforms == 0) {
                LOGGER.warn("[nvidia_chunk] 未找到任何 OpenCL 平台，GPU 加速不可用");
                return null;
            }

            cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
            clGetPlatformIDs(numPlatforms, platforms, null);

            // 收集所有 (platform, device) 候选
            List<PlatformDevice> candidates = new ArrayList<>();
            for (cl_platform_id platform : platforms) {
                String platformName = getString(platform, CL_PLATFORM_NAME);
                String platformVendor = getString(platform, CL_PLATFORM_VENDOR);

                int[] numDevicesArray = new int[1];
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, null, numDevicesArray);
                int numDevices = numDevicesArray[0];
                if (numDevices == 0) continue;

                cl_device_id[] devices = new cl_device_id[numDevices];
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, null);

                for (int i = 0; i < numDevices; i++) {
                    String devName = getDeviceString(devices[i], CL_DEVICE_NAME);
                    String devVendor = getDeviceString(devices[i], CL_DEVICE_VENDOR);
                    candidates.add(new PlatformDevice(platform, platformName, platformVendor,
                            devices[i], i, devName, devVendor));
                }
            }

            if (candidates.isEmpty()) {
                LOGGER.warn("[nvidia_chunk] 未找到任何 OpenCL 设备，GPU 加速不可用");
                return null;
            }

            // 按首选厂商筛选
            String pref = preferredVendor == null ? "auto" : preferredVendor.toLowerCase(Locale.ROOT);
            PlatformDevice chosen = null;

            if (!"auto".equals(pref)) {
                String[] keywords = switch (pref) {
                    case "nvidia" -> VENDOR_KEYWORDS[0];
                    case "amd" -> VENDOR_KEYWORDS[1];
                    case "intel" -> VENDOR_KEYWORDS[2];
                    default -> null;
                };
                if (keywords != null) {
                    // 在首选厂商中找 deviceIndex 个
                    int seen = 0;
                    for (PlatformDevice pd : candidates) {
                        if (matchesVendor(pd, keywords)) {
                            if (seen == deviceIndex) {
                                chosen = pd;
                                break;
                            }
                            seen++;
                        }
                    }
                }
            }

            if (chosen == null) {
                // auto 或首选未命中：按 NVIDIA > AMD > Intel 优先级选首个可用
                LOGGER.info("[nvidia_chunk] 首选厂商 {} 未命中，按优先级自动选择", pref);
                for (int v = 0; v < VENDOR_KEYWORDS.length && chosen == null; v++) {
                    for (PlatformDevice pd : candidates) {
                        if (matchesVendor(pd, VENDOR_KEYWORDS[v])) {
                            chosen = pd;
                            break;
                        }
                    }
                }
            }
            if (chosen == null) {
                // 兜底：选第 0 个
                chosen = candidates.get(0);
            }

            LOGGER.info("[nvidia_chunk] 选定 OpenCL 设备：{} ({}) @ 平台 {}",
                    chosen.deviceName, chosen.deviceVendor, chosen.platformName);

            logDeviceDetails(chosen.device, chosen.platform);

            // 检查双精度支持（MC ImprovedNoise 必须用 double）
            // 注意：CL_DEVICE_DOUBLE_FP_CONFIG 返回 cl_device_fp_config（cl_ulong，8 字节），
            // 不能用 Sizeof.cl_int（4 字节）读取，否则 OpenCL 返回 CL_INVALID_VALUE。
            long[] doubleFpConfig = new long[1];
            clGetDeviceInfo(chosen.device, CL_DEVICE_DOUBLE_FP_CONFIG, Sizeof.cl_ulong,
                    Pointer.to(doubleFpConfig), null);
            if (doubleFpConfig[0] == 0L) {
                LOGGER.warn("[nvidia_chunk] 设备 {} 不支持双精度浮点(cl_khr_fp64)，GPU 加速不可用。" +
                                "请使用支持 cl_khr_fp64 的 NVIDIA / AMD / Intel 独显。",
                        chosen.deviceName);
                return null;
            }

            // 创建上下文
            cl_context_properties ctxProps = new cl_context_properties();
            ctxProps.addProperty(CL_CONTEXT_PLATFORM, chosen.platform);
            cl_context context = clCreateContext(ctxProps, 1, new cl_device_id[]{chosen.device},
                    null, null, null);

            // 编译内核
            cl_program program;
            try {
                program = clCreateProgramWithSource(context, 1, new String[]{KERNEL_SOURCE}, null, null);
                clBuildProgram(program, 1, new cl_device_id[]{chosen.device}, null, null, null);
            } catch (CLException e) {
                // 服务端无 GPU 或驱动不支持 cl_khr_fp64 是常见情况，用 WARN 避免引起管理员恐慌
                LOGGER.warn("[nvidia_chunk] OpenCL 内核编译失败，设备可能不支持 cl_khr_fp64。将使用 CPU 回退。错误：{}", e.getMessage());
                clReleaseContext(context);
                return null;
            }

            cl_kernel kernel = clCreateKernel(program, "improved_noise_kernel", null);

            // 实际工作组大小
            long[] maxWgs = new long[1];
            clGetKernelWorkGroupInfo(kernel, chosen.device, CL_KERNEL_WORK_GROUP_SIZE,
                    Sizeof.size_t, Pointer.to(maxWgs), null);
            int actualWgs = (int) Math.min(workGroupSize, maxWgs[0]);
            if (actualWgs < 1) actualWgs = 1;

            instance = new GPUContext(context, chosen.device, program, kernel, actualWgs,
                    chosen.deviceVendor, chosen.deviceName);
            LOGGER.info("[nvidia_chunk] GPU 上下文初始化成功，工作组大小 = {}", actualWgs);
            return instance;

        } catch (CLException e) {
            // GPU 不可用是可恢复的（有 CPU 回退），用 WARN 而非 ERROR
            LOGGER.warn("[nvidia_chunk] GPU 上下文初始化失败，将使用 CPU 回退：{}", e.getMessage());
            return null;
        } catch (Throwable t) {
            // 兜底任何 JNI / UnsatisfiedLinkError（服务端无 OpenCL 运行时时常见）
            LOGGER.warn("[nvidia_chunk] GPU 初始化异常（可能 JOCL 原生库加载失败，服务端无 OpenCL 运行时），将使用 CPU 回退：{}", t.getMessage());
            return null;
        }
    }

    public static GPUContext getInstance() {
        return instance;
    }

    public boolean isAvailable() {
        return available && !closed;
    }

    public String getDeviceName() {
        return deviceName;
    }

    public String getVendorName() {
        return vendorName;
    }

    public int getWorkGroupSize() {
        return workGroupSize;
    }

    /**
     * 标记 GPU 路径不可用（上层回退到 CPU）。
     */
    public void markUnavailable() {
        this.available = false;
    }

    /**
     * 输出 GPU 设备详细信息，用于调试和确认设备能力。
     */
    private static void logDeviceDetails(cl_device_id device, cl_platform_id platform) {
        try {
            StringBuilder sb = new StringBuilder();
            sb.append("[nvidia_chunk] GPU 设备详情：\n");

            String platformVersion = getString(platform, CL_PLATFORM_VERSION);
            sb.append("[nvidia_chunk]   平台版本: ").append(platformVersion).append("\n");

            String driverVersion = getDeviceString(device, CL_DRIVER_VERSION);
            sb.append("[nvidia_chunk]   驱动版本: ").append(driverVersion).append("\n");

            String deviceVersion = getDeviceString(device, CL_DEVICE_VERSION);
            sb.append("[nvidia_chunk]   OpenCL版本: ").append(deviceVersion).append("\n");

            long[] maxComputeUnits = new long[1];
            clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, Sizeof.cl_uint,
                    Pointer.to(maxComputeUnits), null);
            sb.append("[nvidia_chunk]   计算单元数: ").append(maxComputeUnits[0]).append("\n");

            long[] maxWorkGroupSize = new long[1];
            clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, Sizeof.size_t,
                    Pointer.to(maxWorkGroupSize), null);
            sb.append("[nvidia_chunk]   最大工作组: ").append(maxWorkGroupSize[0]).append("\n");

            long[] maxWorkItemSizes = new long[3];
            clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3 * Sizeof.size_t,
                    Pointer.to(maxWorkItemSizes), null);
            sb.append("[nvidia_chunk]   最大工作项维度: [")
                    .append(maxWorkItemSizes[0]).append(", ")
                    .append(maxWorkItemSizes[1]).append(", ")
                    .append(maxWorkItemSizes[2]).append("]\n");

            long[] globalMemSize = new long[1];
            clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, Sizeof.size_t,
                    Pointer.to(globalMemSize), null);
            sb.append("[nvidia_chunk]   全局内存: ").append(formatBytes(globalMemSize[0])).append("\n");

            long[] localMemSize = new long[1];
            clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, Sizeof.size_t,
                    Pointer.to(localMemSize), null);
            sb.append("[nvidia_chunk]   本地内存: ").append(formatBytes(localMemSize[0])).append("\n");

            long[] maxAllocSize = new long[1];
            clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, Sizeof.size_t,
                    Pointer.to(maxAllocSize), null);
            sb.append("[nvidia_chunk]   最大单块分配: ").append(formatBytes(maxAllocSize[0])).append("\n");

            long[] clockFreq = new long[1];
            clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, Sizeof.cl_uint,
                    Pointer.to(clockFreq), null);
            sb.append("[nvidia_chunk]   时钟频率: ").append(clockFreq[0]).append(" MHz\n");

            long[] memClockFreq = new long[1];
            try {
                clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, Sizeof.cl_uint,
                        Pointer.to(memClockFreq), null);
            } catch (Throwable ignored) {
            }
            sb.append("[nvidia_chunk]   内存对齐: ").append(memClockFreq[0]).append(" bits\n");

            long[] fpConfig = new long[1];
            clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, Sizeof.cl_ulong,
                    Pointer.to(fpConfig), null);
            sb.append("[nvidia_chunk]   双精度配置: ").append(fpConfig[0]).append("\n");

            long[] extensions = new long[1];
            clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, null, extensions);
            byte[] extBuf = new byte[(int) extensions[0]];
            clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, extBuf.length, Pointer.to(extBuf), null);
            String extStr = new String(extBuf).trim();
            sb.append("[nvidia_chunk]   扩展: ").append(extStr.length() > 100 ? extStr.substring(0, 100) + "..." : extStr).append("\n");

            LOGGER.info(sb.toString());
        } catch (Throwable t) {
            LOGGER.warn("[nvidia_chunk] 获取设备详情失败: {}", t.getMessage());
        }
    }

    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
    }

    // ------------------------------------------------------------------
    // 调度接口
    // ------------------------------------------------------------------

    /**
     * 批量调度 ImprovedNoise.noise() 到 GPU。
     *
     * <p>所有输入必须对应同一个 ImprovedNoise 实例（相同的 perm/xo/yo/zo）。</p>
     *
     * <p>性能优化：使用 PerThreadState 中复用的数组（scalarArgs/countArg/globalWorkSize/
     * localWorkSize），避免每次调度创建短生命周期对象。clSetKernelArg 会立即拷贝参数值，
     * 因此复用数组安全。</p>
     *
     * <p>同步语义：{@code clEnqueueReadBuffer(CL_TRUE)} 是阻塞读取，会等待该读操作之前的
     * 所有命令完成。因此无需额外调用 {@code clFinish}。</p>
     *
     * @param perm    256 字节置换表
     * @param xo      x 偏移
     * @param yo      y 偏移
     * @param zo      z 偏移
     * @param inputs  count*5 个 double：[x0,y0,z0,yff0,yfv0, x1,y1,...]
     * @param count   输入项数
     * @param outputs count 个 double 的输出数组（由调用方预分配）
     */
    public void dispatchBatch(byte[] perm, double xo, double yo, double zo,
                              double[] inputs, int count, double[] outputs) {
        if (!isAvailable()) {
            throw new IllegalStateException("GPU 不可用");
        }
        if (count <= 0) return;

        PerThreadState state = getPerThreadState();
        try {
            state.ensureBuffers(count);
            state.uploadPerm(perm);
            state.uploadInputs(inputs, count);

            // 复用 scalarArgs[0] 逐个设置 xo/yo/zo（clSetKernelArg 立即拷贝，安全）
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(state.permMem));
            state.scalarArgs[0] = xo;
            clSetKernelArg(kernel, 1, Sizeof.cl_double, Pointer.to(state.scalarArgs));
            state.scalarArgs[0] = yo;
            clSetKernelArg(kernel, 2, Sizeof.cl_double, Pointer.to(state.scalarArgs));
            state.scalarArgs[0] = zo;
            clSetKernelArg(kernel, 3, Sizeof.cl_double, Pointer.to(state.scalarArgs));
            clSetKernelArg(kernel, 4, Sizeof.cl_mem, Pointer.to(state.inputMem));
            clSetKernelArg(kernel, 5, Sizeof.cl_mem, Pointer.to(state.outputMem));
            state.countArg[0] = count;
            clSetKernelArg(kernel, 6, Sizeof.cl_int, Pointer.to(state.countArg));

            // 全局大小：向上对齐到 workGroupSize（复用数组）
            long global = ((count + workGroupSize - 1) / workGroupSize) * (long) workGroupSize;
            state.globalWorkSize[0] = global;
            // localWorkSize 在构造函数中已设置为 workGroupSize，无需重复赋值

            clEnqueueNDRangeKernel(state.queue, kernel, 1, null,
                    state.globalWorkSize, state.localWorkSize, 0, null, null);
            // CL_TRUE 阻塞读取：等待内核完成，无需额外 clFinish
            clEnqueueReadBuffer(state.queue, state.outputMem, CL_TRUE, 0,
                    (long) count * Sizeof.cl_double, Pointer.to(outputs), 0, null, null);

        } catch (CLException e) {
            if (Config.verbose) {
                LOGGER.error("[nvidia_chunk] GPU dispatchBatch 失败：{}", e.getMessage(), e);
            }
            markUnavailable();
            throw e;
        }
    }

    /**
     * 单次调度 ImprovedNoise.noise()。比批量慢（每次有完整启动开销），
     * 仅用于缓存未命中且无法批量时。
     *
     * <p>性能优化：使用 PerThreadState 中复用的 singleInputs/singleOutputs 数组，
     * 避免每次调用创建 new double[5] 和 new double[1]。</p>
     */
    public double dispatchSingle(byte[] perm, double xo, double yo, double zo,
                                 double x, double y, double z,
                                 double yFloorFreq, double yFloorValue) {
        PerThreadState state = getPerThreadState();
        state.singleInputs[0] = x;
        state.singleInputs[1] = y;
        state.singleInputs[2] = z;
        state.singleInputs[3] = yFloorFreq;
        state.singleInputs[4] = yFloorValue;
        dispatchBatch(perm, xo, yo, zo, state.singleInputs, 1, state.singleOutputs);
        return state.singleOutputs[0];
    }

    // ------------------------------------------------------------------
    // 每线程状态
    // ------------------------------------------------------------------

    private PerThreadState getPerThreadState() {
        PerThreadState state = perThread.get();
        if (state == null) {
            state = new PerThreadState(context, device, workGroupSize);
            perThread.set(state);
            startCleanupThread();
            StateReference ref = new StateReference(state, cleanupQueue);
            trackedStates.add(ref);
        }
        return state;
    }

    /**
     * 每线程持有：命令队列 + 可重用 cl_mem 缓冲区 + Java 端 double 数组。
     *
     * <p>perm 缓冲区在 {@link #uploadPerm} 中按需重传（通过哈希比较避免重复传输）。</p>
     *
     * <p>性能优化：所有临时数组（singleInputs/singleOutputs/scalarArgs/countArg/
     * globalWorkSize/localWorkSize）均为每线程复用，避免每次 dispatchSingle/dispatchBatch
     * 创建短生命周期对象，显著降低 GC 压力。</p>
     */
    private static final class PerThreadState {
        final cl_context context;
        final cl_command_queue queue;
        final int workGroupSize;

        cl_mem permMem;
        cl_mem inputMem;
        cl_mem outputMem;
        int inputCapacity;   // 当前 inputMem/outputMem 可容纳的项数
        int cachedPermHash;  // 已上传 perm 的哈希，避免重复传输
        byte[] cachedPerm;   // 已上传的 perm 副本
        byte[] cachedPermRef; // 已上传 perm 的引用（O(1) 引用比较快速路径）

        // 复用数组：避免每次调度创建临时对象
        final double[] singleInputs;   // dispatchSingle 的 5 元素输入
        final double[] singleOutputs;  // dispatchSingle 的 1 元素输出
        final double[] scalarArgs;     // xo/yo/zo 三个标量参数
        final int[] countArg;          // count 参数
        final long[] globalWorkSize;   // 全局工作大小
        final long[] localWorkSize;    // 局部工作大小

        PerThreadState(cl_context context, cl_device_id device, int workGroupSize) {
            this.context = context;
            // 使用 clCreateCommandQueue（OpenCL 1.x 兼容，避免 cl_queue_properties 在某些 JOCL 版本缺失）
            this.queue = clCreateCommandQueue(context, device, 0L, null);
            this.workGroupSize = workGroupSize;
            // perm 缓冲区固定 256 字节，预先创建（CL_MEM_COPY_HOST_PTR 需要非 null 主机指针）
            this.permMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    256L, Pointer.to(new byte[256]), null);
            this.cachedPermHash = 0;
            this.cachedPerm = new byte[256];
            this.inputCapacity = 0;
            // 预分配默认容量（64 项），避免首次调度即触发扩容
            ensureBuffers(64);
            // 初始化复用数组
            this.singleInputs = new double[5];
            this.singleOutputs = new double[1];
            this.scalarArgs = new double[3];
            this.countArg = new int[1];
            this.globalWorkSize = new long[1];
            this.localWorkSize = new long[]{workGroupSize};
        }

        void ensureBuffers(int requiredCount) {
            if (requiredCount <= inputCapacity) return;
            // 释放旧缓冲区
            if (inputMem != null) clReleaseMemObject(inputMem);
            if (outputMem != null) clReleaseMemObject(outputMem);
            // 按 2 倍向上扩容，至少 64
            int newCap = Math.max(64, Integer.highestOneBit(requiredCount - 1) << 2);
            long inputBytes = (long) newCap * 5L * Sizeof.cl_double;
            long outputBytes = (long) newCap * Sizeof.cl_double;
            this.inputMem = clCreateBuffer(context, CL_MEM_READ_ONLY, inputBytes, null, null);
            this.outputMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, outputBytes, null, null);
            this.inputCapacity = newCap;
        }

        void uploadPerm(byte[] perm) {
            // 引用比较快速路径：ImprovedNoise.p 是 final 字段，同一实例的 perm 引用不变。
            // 同一实例连续调用 dispatchBatch 时，这里 O(1) 直接返回，省去 O(256) 的
            // Arrays.hashCode + Arrays.equals。
            if (perm == cachedPermRef) return;
            int hash = Arrays.hashCode(perm);
            if (hash == cachedPermHash && Arrays.equals(cachedPerm, perm)) {
                cachedPermRef = perm;
                return;
            }
            ByteBuffer bb = ByteBuffer.allocate(256).order(ByteOrder.nativeOrder());
            bb.put(perm);
            bb.flip();
            clEnqueueWriteBuffer(queue, permMem, CL_TRUE, 0, 256,
                    Pointer.to(bb.array()), 0, null, null);
            System.arraycopy(perm, 0, cachedPerm, 0, 256);
            cachedPermHash = hash;
            cachedPermRef = perm;
        }

        void uploadInputs(double[] inputs, int count) {
            clEnqueueWriteBuffer(queue, inputMem, CL_TRUE, 0,
                    (long) count * 5L * Sizeof.cl_double,
                    Pointer.to(inputs), 0, null, null);
        }
    }

    // ------------------------------------------------------------------
    // 关闭
    // ------------------------------------------------------------------

    /**
     * 释放所有 OpenCL 资源。仅应在模组卸载 / 游戏关闭时调用。
     */
    public synchronized void close() {
        if (closed) return;
        closed = true;
        available = false;
        try {
            if (cleanupThread != null) {
                cleanupThread.interrupt();
                try {
                    cleanupThread.join(5000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                cleanupThread = null;
            }

            for (StateReference ref : trackedStates) {
                releasePerThreadState(ref.state);
            }
            trackedStates.clear();

            PerThreadState state = perThread.get();
            if (state != null) {
                releasePerThreadState(state);
                perThread.remove();
            }
            if (kernel != null) clReleaseKernel(kernel);
            if (program != null) clReleaseProgram(program);
            if (device != null) clReleaseDevice(device);
            if (context != null) clReleaseContext(context);
            LOGGER.info("[nvidia_chunk] GPU 上下文已释放");
        } catch (Throwable t) {
            LOGGER.warn("[nvidia_chunk] GPU 资源释放异常：{}", t.getMessage());
        }
    }

    private static void releasePerThreadState(PerThreadState state) {
        try {
            if (state.inputMem != null) clReleaseMemObject(state.inputMem);
            if (state.outputMem != null) clReleaseMemObject(state.outputMem);
            if (state.permMem != null) clReleaseMemObject(state.permMem);
            if (state.queue != null) clReleaseCommandQueue(state.queue);
        } catch (Throwable ignored) {
        }
    }

    // ------------------------------------------------------------------
    // 工具方法
    // ------------------------------------------------------------------

    private static boolean matchesVendor(PlatformDevice pd, String[] keywords) {
        String combined = (pd.platformName + " " + pd.platformVendor + " "
                + pd.deviceName + " " + pd.deviceVendor).toLowerCase(Locale.ROOT);
        for (String kw : keywords) {
            if (combined.contains(kw.toLowerCase(Locale.ROOT))) return true;
        }
        return false;
    }

    private static String getString(cl_platform_id platform, int paramName) {
        long[] size = new long[1];
        clGetPlatformInfo(platform, paramName, 0, null, size);
        byte[] buf = new byte[(int) size[0]];
        clGetPlatformInfo(platform, paramName, buf.length, Pointer.to(buf), null);
        return new String(buf, 0, buf.length).trim();
    }

    private static String getDeviceString(cl_device_id device, int paramName) {
        long[] size = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);
        byte[] buf = new byte[(int) size[0]];
        clGetDeviceInfo(device, paramName, buf.length, Pointer.to(buf), null);
        return new String(buf, 0, buf.length).trim();
    }

    private static final class PlatformDevice {
        final cl_platform_id platform;
        final String platformName;
        final String platformVendor;
        final cl_device_id device;
        final int deviceIndex;
        final String deviceName;
        final String deviceVendor;

        PlatformDevice(cl_platform_id platform, String platformName, String platformVendor,
                       cl_device_id device, int deviceIndex, String deviceName, String deviceVendor) {
            this.platform = platform;
            this.platformName = platformName;
            this.platformVendor = platformVendor;
            this.device = device;
            this.deviceIndex = deviceIndex;
            this.deviceName = deviceName;
            this.deviceVendor = deviceVendor;
        }
    }
}
