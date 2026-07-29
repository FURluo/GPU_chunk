package furluo.nvidia_chunk.gpu;

import com.mojang.logging.LogUtils;
import furluo.nvidia_chunk.Config;
import net.minecraft.world.level.levelgen.synth.ImprovedNoise;
import org.slf4j.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * GPU 噪声调度管理器（单例）。
 *
 * <p>职责：</p>
 * <ul>
 *   <li>对每次 {@link ImprovedNoise#noise(double, double, double, double, double)} 调用，
 *       先查 {@link NoiseCache}；命中直接返回。</li>
 *   <li>未命中时：GPU 可用则单次调度 GPU，不可用则走 {@link #cpuNoise} Java 回退实现。</li>
 *   <li>无论 GPU 还是 CPU 计算，结果都写入缓存，后续相同坐标命中。</li>
 *   <li>跟踪连续 GPU 失败次数，超过 {@link Config#failuresBeforeDisable} 后自动禁用 GPU。</li>
 * </ul>
 *
 * <p>设计说明：本管理器始终"消费"掉 noise 调用（不回退到原版方法），因此 Mixin 层使用
 * {@code @Inject(cancellable=true)} 在 HEAD 处直接 setReturnValue。
 * 这样做的好处是缓存对 GPU 与 CPU 路径都生效，命中率有保证。</p>
 *
 * <p><b>安全性保证</b>：所有公开方法（{@link #dispatchNoise}、{@link #prefetchBatch}）
 * 都对参数进行校验。{@code dispatchNoise} 若检测到异常 perm 表会抛出
 * {@link IllegalArgumentException}，由 Mixin 层的 try-catch 捕获并回退原版逻辑，
 * 确保不崩溃。{@code prefetchBatch} 整体 try-catch 包裹，预取失败完全静默，
 * 不影响主路径。</p>
 *
 * <p>正确性保证：{@link #cpuNoise} 与 OpenCL 内核 {@code improved_noise_kernel} 均严格按
 * MC 1.20.1 ImprovedNoise 字节码移植，输出与原版逐位一致。</p>
 */
public final class GPUNoiseManager {

    private static final Logger LOGGER = LogUtils.getLogger();

    private static final double SHIFT_UP_EPSILON = 1.0000000116860974E-7; // (double)1.0E-7f

    /** 单次 GPU 预取批量上限（坐标数）。init 提示与 speculativePrefetch 截断共用同一常数。 */
    static final int MAX_SAFE_PREFETCH = 512;

    /**
     * GRADIENT 表，展平为一维数组以提升缓存局部性。
     * 与 SimplexNoise.GRADIENT 完全一致：GRADIENT[hash & 15] = {gx, gy, gz}。
     * 布局：[g0x, g0y, g0z, g1x, g1y, g1z, ...]
     */
    private static final int[] GRADIENT_FLAT = new int[] {
             1,  1, 0,    -1,  1, 0,     1, -1, 0,    -1, -1, 0,
             1,  0, 1,    -1,  0, 1,     1,  0, -1,   -1,  0, -1,
             0,  1, 1,     0, -1, 1,     0,  1, -1,    0, -1, -1,
             1,  1, 0,     0, -1, 1,    -1,  1, 0,     0, -1, -1
    };

    private static volatile GPUNoiseManager instance;

    private final GPUContext gpu;
    private volatile boolean gpuDisabled;        // 累计失败后永久禁用
    private final AtomicInteger consecutiveFailures = new AtomicInteger();
    private volatile boolean loggedOnce;

    // ------------------------------------------------------------------
    // 运行时统计：用于确认 GPU 是否被调用、命中率、预取效果等
    // ------------------------------------------------------------------
    // LongAdder：热路径每次 noise() 调用都要计数，AtomicLong 在多工作线程下
    // 会产生跨核缓存行争用，LongAdder 分片累加避免争用
    private final LongAdder totalCalls = new LongAdder();
    private final LongAdder cacheHits = new LongAdder();
    private final LongAdder cpuComputations = new LongAdder();
    private final LongAdder gpuBatchDispatches = new LongAdder();
    private final LongAdder gpuBatchComputations = new LongAdder();
    private final LongAdder prefetchTriggers = new LongAdder();
    private final LongAdder prefetchSuccesses = new LongAdder();
    private final LongAdder prefetchFails = new LongAdder();
    private final LongAdder prefetchPredictedCoords = new LongAdder();
    private final LongAdder chunksSeen = new LongAdder();
    private volatile long lastStatsLogTime;
    private volatile boolean loggedFirstChunk;
    private volatile boolean loggedPrefetchDiagnostic;

    // 跨 chunk 缓存复用：每 N 个 chunk 清空一次缓存
    private static final int CHUNKS_BEFORE_CLEAR = 4;
    private int chunkCounter = 0;

    // 每线程线程诊断标记（避免 verbose 模式下每次调用都执行 LongAdder.sum()）
    private final ThreadLocal<Boolean> threadDiagLogged = ThreadLocal.withInitial(() -> Boolean.FALSE);

    // ------------------------------------------------------------------
    // 错误日志去重：避免 dispatchNoise 异常时日志刷屏
    // ------------------------------------------------------------------
    private static final AtomicLong suppressedErrorCount = new AtomicLong();

    /**
     * 记录被 Mixin 层吞掉的异常（dispatchNoise 抛出的异常）。
     *
     * <p>日志策略：</p>
     * <ul>
     *   <li>第 1 次：记录 WARN + 异常摘要（verbose 时附带堆栈）</li>
     *   <li>后续每 1000 次：记录一次 WARN 摘要</li>
     *   <li>其余静默（Mixin 已自动回退原版，不影响游戏）</li>
     * </ul>
     */
    public static void logSuppressedError(Throwable t) {
        long count = suppressedErrorCount.incrementAndGet();
        if (count == 1) {
            LOGGER.warn("[nvidia_chunk] dispatchNoise 首次异常，已回退原版逻辑（不崩溃）。原因：{}",
                    t.toString());
            if (Config.verbose) {
                LOGGER.warn("[nvidia_chunk] 异常堆栈：", t);
            }
        } else if (count % 1000 == 0) {
            LOGGER.warn("[nvidia_chunk] dispatchNoise 已累计异常 {} 次（已自动回退原版），最近原因：{}",
                    count, t.toString());
        }
    }

    private GPUNoiseManager(GPUContext gpu) {
        this.gpu = gpu;
    }

    /**
     * 初始化管理器。应在模组加载、GPU 上下文创建后调用。
     */
    public static synchronized void init(GPUContext gpu) {
        if (instance != null) {
            LOGGER.warn("[nvidia_chunk] GPUNoiseManager 已初始化");
            return;
        }
        instance = new GPUNoiseManager(gpu);

        // 配置验证：强制检查配置值是否在安全范围内
        int configuredPrefetchSize = Config.PREFETCH_SIZE_INT;
        if (configuredPrefetchSize > MAX_SAFE_PREFETCH) {
            LOGGER.warn("[nvidia_chunk] 配置的 prefetchSize={} 超过安全上限 {}，运行时将自动截断",
                    configuredPrefetchSize, MAX_SAFE_PREFETCH);
        }
        if (configuredPrefetchSize > 0) {
            LOGGER.warn("[nvidia_chunk] prefetchSize={} > 0：GPU 预取在持续区块生成中实测利用率仅 0.06%，",
                    configuredPrefetchSize);
            LOGGER.warn("[nvidia_chunk]   原因：ImprovedNoise 实例频繁切换导致步长预测失效，预取坐标几乎全部无效。");
            LOGGER.warn("[nvidia_chunk]   建议：设置 prefetchSize=0 可消除每次缓存未命中 ~50ns 的预取检查开销，提升 TPS。");
        }

        LOGGER.info("[nvidia_chunk] 配置确认：prefetchSize={}, verbose={}, GPU启用={}",
                configuredPrefetchSize, Config.verbose, Config.enabled);

        if (gpu != null && gpu.isAvailable()) {
            LOGGER.info("[nvidia_chunk] GPUNoiseManager 已启用，GPU = {} ({})",
                    gpu.getDeviceName(), gpu.getVendorName());
        } else {
            LOGGER.info("[nvidia_chunk] GPUNoiseManager 已启用，GPU 不可用，将使用 CPU 回退（仍享受缓存加速）");
        }
    }

    public static GPUNoiseManager getInstance() {
        return instance;
    }

    /**
     * 处理一次 ImprovedNoise.noise(x, y, z, yFloorFreq, yFloorValue) 调用。
     *
     * <p>由 {@code ImprovedNoiseMixin} 在 HEAD 处调用，返回值直接作为原方法返回值。</p>
     *
     * <p>性能策略（v2 优化）：</p>
     * <ul>
     *   <li>缓存命中（~87%）：直接返回，零计算开销。</li>
     *   <li>缓存未命中：默认走 CPU 回退（~100ns），因为 GPU 单次调度开销
     *       （~50-200μs，含内核启动 + 同步读回）远高于 CPU 计算。
     *       仅当 {@link Config#useGpuForSingleDispatch} 显式启用时才走 GPU 单次调度
     *       （用于正确性验证或特殊场景）。</li>
     *   <li>真正的 GPU 加速通过 {@link #prefetchBatch} 批量预取实现：
     *       在 NoiseChunk 构造时一次性提交数千个坐标到 GPU，预填缓存。</li>
     * </ul>
     *
     * <p><b>安全性</b>：若 perm 表无效（null 或长度 &lt; 256），抛出
     * {@link IllegalArgumentException}，由 Mixin 层捕获并回退原版逻辑。</p>
     *
     * @param instance    目标 ImprovedNoise 实例（提供 perm 表与 xo/yo/zo 偏移）
     * @param perm        实例的置换表（256 字节，由 Mixin @Shadow 读取）
     * @param xo yo zo    实例的坐标偏移
     */
    public double dispatchNoise(ImprovedNoise instance, byte[] perm,
                                double xo, double yo, double zo,
                                double x, double y, double z,
                                double yFloorFreq, double yFloorValue) {
        totalCalls.increment();

        // 线程诊断：每线程仅首次调用时记录（修复：旧代码每次调用都执行 totalCalls.sum()）
        if (Config.verbose && !threadDiagLogged.get()) {
            threadDiagLogged.set(Boolean.TRUE);
            Thread currentThread = Thread.currentThread();
            LOGGER.info("[nvidia_chunk] [线程诊断] dispatchNoise 在线程 {} 执行 (id={})",
                    currentThread.getName(), currentThread.getId());
        }

        // 1) 缓存命中（新接口返回 double，NaN 表示未命中，无 Double 装箱）
        //
        //    perm 校验移到未命中后：87% 命中路径省去 perm 校验开销。
        //    安全性分析：cache 的 key 是 (instanceId, x, y, z, yff, yfv)，
        //    不含 perm。若 perm 被反射修改，cache 会返回旧 perm 的结果，
        //    但 ImprovedNoise.p 是 final 字段，正常情况不变；且即使返回
        //    旧结果也只是区块生成略有偏差，不会崩溃。Mixin 层 try-catch
        //    仍能捕获未命中路径的 perm 校验异常。
        NoiseCache cache = NoiseCache.get();
        double cached = cache.get(instance, x, y, z, yFloorFreq, yFloorValue);
        if (!Double.isNaN(cached)) {
            cacheHits.increment();
            return cached;
        }

        // 2) 缓存未命中：先校验 perm 表（防御性检查）
        //    perm 表必须有效（256 字节），否则 cpuNoise/gpu 调度会数组越界
        if (perm == null || perm.length < 256) {
            throw new IllegalArgumentException(
                    "perm table is null or too short: "
                            + (perm == null ? "null" : String.valueOf(perm.length)));
        }

        // 3) CPU/GPU 计算（默认 CPU，仅在显式启用时走 GPU 单次调度）
        double result;
        boolean useGpu = Config.enabled
                && Config.useGpuForSingleDispatch
                && gpu != null
                && gpu.isAvailable()
                && !gpuDisabled;
        if (useGpu) {
            try {
                result = gpu.dispatchSingle(perm, xo, yo, zo, x, y, z, yFloorFreq, yFloorValue);
                consecutiveFailures.set(0);
            } catch (Throwable t) {
                onGpuFailure(t);
                cpuComputations.increment();
                result = cpuNoise(perm, xo, yo, zo, x, y, z, yFloorFreq, yFloorValue);
            }
        } else {
            cpuComputations.increment();
            result = cpuNoise(perm, xo, yo, zo, x, y, z, yFloorFreq, yFloorValue);
        }

        // 3) 写入缓存（try-catch 保护：缓存写入失败不影响返回值）
        try {
            cache.put(instance, x, y, z, yFloorFreq, yFloorValue, result);
        } catch (Throwable t) {
            // 缓存写入异常（如扩容 OOM）：静默忽略，下次重新计算
            if (Config.verbose) {
                LOGGER.warn("[nvidia_chunk] 缓存写入异常（已忽略）：{}", t.toString());
            }
        }

        // 4) 自适应投机预取：观察连续未命中坐标的差值，推测 cell 的 X/Z 步长；
        //    步长稳定后预测「尚未计算」的相邻坐标，批量提交 GPU 预填缓存。
        //    （修复：旧实现把"刚刚已被 CPU 计算并写入缓存"的坐标累加进批次，
        //    flush 时又按缓存过滤，批次永远为空，GPU 从未被真正调度。）
        //
        //    触发策略：
        //    - 步长稳定后首次未命中：立即预取一次，快速预填当前 cell 附近顶点。
        //    - 后续未命中：每 prefetchSize 次未命中触发一次，平衡命中率与调度开销。
        //    预取完全隔离：任何异常都不影响主路径（result 已计算完成）。
        if (Config.enabled
                && Config.PREFETCH_SIZE_INT > 0
                && gpu != null
                && gpu.isAvailable()
                && !gpuDisabled) {
            try {
                PrefetchBuffers buffers = prefetchBuffers.get();
                InstanceStepState state = findOrCreateInstanceState(instance, buffers);
                observeStep(instance, x, y, z, state);
                if (state.stepStable && !state.alreadyPrefetched) {
                    state.alreadyPrefetched = true;
                    prefetchTriggers.increment();
                    speculativePrefetch(instance, perm, xo, yo, zo,
                            x, y, z, yFloorFreq, yFloorValue, state, buffers);
                }
            } catch (Throwable t) {
                prefetchFails.increment();
                if (Config.verbose) {
                    LOGGER.warn("[nvidia_chunk] 预取异常（已忽略）：{}", t.toString());
                }
            }
        }

        return result;
    }

    /**
     * 查找或创建当前 ImprovedNoise 实例的步长状态。
     * 每个实例独立跟踪，不受其他实例切换影响。
     */
    private InstanceStepState findOrCreateInstanceState(ImprovedNoise instance, PrefetchBuffers buffers) {
        for (int i = 0; i < buffers.instanceCount; i++) {
            if (buffers.instanceStates[i].instance == instance) {
                return buffers.instanceStates[i];
            }
        }
        if (buffers.instanceCount == buffers.instanceStates.length) {
            InstanceStepState[] newArr = new InstanceStepState[buffers.instanceStates.length * 2];
            System.arraycopy(buffers.instanceStates, 0, newArr, 0, buffers.instanceStates.length);
            buffers.instanceStates = newArr;
        }
        InstanceStepState state = new InstanceStepState();
        state.instance = instance;
        buffers.instanceStates[buffers.instanceCount++] = state;
        if (Config.verbose) {
            LOGGER.info("[nvidia_chunk] [调试] 新 ImprovedNoise 实例登记 #{}, 已跟踪实例数={}",
                    buffers.instanceCount, buffers.instanceCount);
        }
        return state;
    }

    /**
     * 观察连续未命中坐标的差值，自适应推测 cell X/Z 步长。
     * 每个实例独立观察，不受其他实例切换影响。
     */
    private void observeStep(ImprovedNoise instance, double x, double y, double z,
                             InstanceStepState state) {
        if (Double.isNaN(state.lastX)) {
            state.lastX = x;
            state.lastY = y;
            state.lastZ = z;
            state.observations = 1;
            return;
        }

        double dx = x - state.lastX;
        double dy = y - state.lastY;
        double dz = z - state.lastZ;

        state.lastX = x;
        state.lastY = y;
        state.lastZ = z;
        state.observations++;

        if (dx == 0 && dy == 0 && dz == 0) return;

        // 更新 X 步长（取出现过的最大绝对差值）
        if (dx != 0) {
            if (state.stepX == 0) {
                state.stepX = dx;
            } else {
                double absDx = Math.abs(dx);
                double absStep = Math.abs(state.stepX);
                if (absDx > absStep) {
                    state.stepX = dx;
                }
            }
        }

        // 更新 Z 步长
        if (dz != 0) {
            if (state.stepZ == 0) {
                state.stepZ = dz;
            } else {
                double absDz = Math.abs(dz);
                double absStep = Math.abs(state.stepZ);
                if (absDz > absStep) {
                    state.stepZ = dz;
                }
            }
        }

        // 步长稳定判断：X 和 Z 步长都已观察到，且观察次数 >= 4
        if (!state.stepStable && state.stepX != 0 && state.stepZ != 0 && state.observations >= 4) {
            state.stepStable = true;
            if (Config.verbose) {
                LOGGER.info("[nvidia_chunk] [调试] 实例步长稳定！观察次数={}, X步长={}, Z步长={}",
                        state.observations, state.stepX, state.stepZ);
            }
        }
    }

    /**
     * 实例级批量预取：用该实例观察到的 cell 步长预测相邻坐标，批量提交 GPU。
     * 预测策略：以当前坐标为中心，沿 X/Z 轴网格化预测 N 个坐标。
     * 每个实例每缓存周期仅触发一次。
     */
    private void speculativePrefetch(ImprovedNoise instance, byte[] perm,
                                     double xo, double yo, double zo,
                                     double x, double y, double z,
                                     double yFloorFreq, double yFloorValue,
                                     InstanceStepState state, PrefetchBuffers buffers) {
        int n = Config.PREFETCH_SIZE_INT;
        if (n <= 0) return;
        if (n > MAX_SAFE_PREFETCH) n = MAX_SAFE_PREFETCH;

        if (buffers.coords.length < n * 5) {
            buffers.coords = new double[Math.max(n * 5, buffers.coords.length * 2)];
        }

        double sx = state.stepX;
        double sz = state.stepZ;

        int idx = 0;
        // 网格化预测：以 (x, z) 为中心，步长 (sx, sz) 的 gridSize×gridSize 网格
        int gridSize = (int) Math.ceil(Math.sqrt(n));
        int half = gridSize / 2;

        for (int i = -half; i <= half && idx < n; i++) {
            for (int k = -half; k <= half && idx < n; k++) {
                if (i == 0 && k == 0) continue;
                int off = idx * 5;
                buffers.coords[off]     = x + i * sx;
                buffers.coords[off + 1] = y;
                buffers.coords[off + 2] = z + k * sz;
                buffers.coords[off + 3] = yFloorFreq;
                buffers.coords[off + 4] = yFloorValue;
                idx++;
            }
        }

        if (idx == 0) return;

        if (Config.verbose) {
            LOGGER.info("[nvidia_chunk] [调试] 实例预取：预测 {} 个坐标，步长=({},{})",
                    idx, sx, sz);
        }

        prefetchBatch(instance, perm, xo, yo, zo, buffers.coords, idx);
    }

    /**
     * 批量预取：将一组坐标一次性提交到 GPU 计算，结果写入缓存。
     *
     * <p>这是真正利用 GPU 并行能力的接口。应在 NoiseChunk 构造时（或 fillSlice 前）
     * 调用，预填缓存，使后续 {@link #dispatchNoise} 调用直接命中缓存。</p>
     *
     * <p>所有坐标必须对应同一 ImprovedNoise 实例（相同的 perm/xo/yo/zo）。
     * 已在缓存中的坐标会跳过，避免重复计算。</p>
     *
     * <p><b>安全性</b>：整个方法 try-catch 包裹，预取失败完全静默，不影响主路径。
     * 即使 GPU 调度失败、缓存写入失败、缓冲区溢出，都不会影响 {@link #dispatchNoise}
     * 已返回的结果。</p>
     *
     * @param instance    目标 ImprovedNoise 实例
     * @param perm        实例的置换表（256 字节）
     * @param xo yo zo    实例的坐标偏移
     * @param coords      count*5 个 double：[x0,y0,z0,yff0,yfv0, x1,y1,...]
     * @param count       坐标项数
     * @return 实际提交到 GPU 计算的项数（跳过缓存命中的）；失败返回 0
     */
    public int prefetchBatch(ImprovedNoise instance, byte[] perm,
                             double xo, double yo, double zo,
                             double[] coords, int count) {
        // 整体 try-catch：预取失败完全静默，不影响主路径
        try {
            if (!Config.enabled || gpu == null || !gpu.isAvailable() || gpuDisabled || count <= 0) {
                // 修复：旧实现在 gpu == null 时仍调用 gpu.isAvailable()，verbose 模式下 NPE
                if (Config.verbose) {
                    if (gpu == null) {
                        LOGGER.info("[nvidia_chunk] [调试] prefetchBatch 跳过：GPU 为 null");
                    } else if (gpuDisabled) {
                        LOGGER.info("[nvidia_chunk] [调试] prefetchBatch 跳过：GPU 已被禁用");
                    } else if (!gpu.isAvailable()) {
                        LOGGER.info("[nvidia_chunk] [调试] prefetchBatch 跳过：GPU 不可用");
                    }
                }
                return 0;
            }
            if (perm == null || perm.length < 256 || coords == null || coords.length < count * 5) {
                if (Config.verbose && perm == null) {
                    LOGGER.info("[nvidia_chunk] [调试] prefetchBatch 跳过：perm 为 null");
                }
                if (Config.verbose && perm != null && perm.length < 256) {
                    LOGGER.info("[nvidia_chunk] [调试] prefetchBatch 跳过：perm 长度不足 ({})", perm.length);
                }
                return 0;
            }

            long startTime = System.nanoTime();
            NoiseCache cache = NoiseCache.get();

            // 筛选出缓存未命中的坐标
            // 注意：使用 filteredCoords 而非 coords，避免当输入 coords 就是 buffers.coords 时
            // （例如 speculativePrefetch 调用 prefetchBatch 的场景）System.arraycopy 出现读写重叠。
            PrefetchBuffers buffers = prefetchBuffers.get();
            if (buffers.filteredCoords.length < count * 5) {
                buffers.filteredCoords = new double[Math.max(count * 5, buffers.filteredCoords.length * 2)];
            }
            int filtered = 0;
            for (int i = 0; i < count; i++) {
                int off = i * 5;
                double x = coords[off];
                double y = coords[off + 1];
                double z = coords[off + 2];
                double yff = coords[off + 3];
                double yfv = coords[off + 4];
                double cached = cache.get(instance, x, y, z, yff, yfv);
                if (Double.isNaN(cached)) {
                    System.arraycopy(coords, off, buffers.filteredCoords, filtered * 5, 5);
                    filtered++;
                }
            }

            if (filtered == 0) {
                if (Config.verbose) {
                    long elapsed = System.nanoTime() - startTime;
                    LOGGER.info("[nvidia_chunk] [调试] prefetchBatch 跳过：所有 {} 个坐标已在缓存中 (耗时={}μs)",
                            count, elapsed / 1000);
                }
                return 0;
            }

            if (Config.verbose) {
                LOGGER.info("[nvidia_chunk] [调试] prefetchBatch 准备提交：输入={}, 过滤后={}, xo={}, yo={}, zo={}",
                        count, filtered, xo, yo, zo);
            }
            prefetchPredictedCoords.add(filtered);

            // 批量提交 GPU
            if (buffers.results.length < filtered) {
                buffers.results = new double[Math.max(filtered, buffers.results.length * 2)];
            }
            long gpuStartTime = System.nanoTime();
            boolean gpuSuccess = false;
            try {
                gpu.dispatchBatch(perm, xo, yo, zo, buffers.filteredCoords, filtered, buffers.results);
                gpuBatchDispatches.increment();
                gpuBatchComputations.add(filtered);
                consecutiveFailures.set(0);
                gpuSuccess = true;
            } catch (Throwable t) {
                onGpuFailure(t);
                // GPU 失败：CPU 回退逐个计算
                cpuComputations.add(filtered);
                for (int i = 0; i < filtered; i++) {
                    int off = i * 5;
                    buffers.results[i] = cpuNoise(perm, xo, yo, zo,
                            buffers.filteredCoords[off], buffers.filteredCoords[off + 1],
                            buffers.filteredCoords[off + 2],
                            buffers.filteredCoords[off + 3], buffers.filteredCoords[off + 4]);
                }
            }
            long gpuElapsed = System.nanoTime() - gpuStartTime;

            // 写入缓存（try-catch 保护）
            try {
                cache.putBatch(instance, buffers.filteredCoords, buffers.results, filtered);
            } catch (Throwable t) {
                if (Config.verbose) {
                    LOGGER.warn("[nvidia_chunk] 预取结果写入缓存异常（已忽略）：{}", t.toString());
                }
            }

            long totalElapsed = System.nanoTime() - startTime;
            if (Config.verbose || gpuBatchDispatches.sum() <= 5) {
                LOGGER.info("[nvidia_chunk] [调试] prefetchBatch 完成：GPU={}, 计算数={}, GPU耗时={}μs, 总耗时={}μs",
                        gpuSuccess ? "成功" : "失败(回退CPU)", filtered,
                        gpuElapsed / 1000, totalElapsed / 1000);
            }

            prefetchSuccesses.increment();
            return filtered;
        } catch (Throwable t) {
            // 最外层兜底：任何未预期的异常都静默
            if (Config.verbose) {
                LOGGER.warn("[nvidia_chunk] prefetchBatch 整体异常（已忽略）：{}", t.toString());
            }
            return 0;
        }
    }

    /**
     * 每个 ImprovedNoise 实例的独立步长观察状态。
     * 核心改进：不同实例的步长互不干扰，修复实例切换导致步长重置的问题。
     */
    private static final class InstanceStepState {
        ImprovedNoise instance;
        double lastX = Double.NaN;
        double lastY, lastZ;
        double stepX, stepZ;
        int observations;
        boolean stepStable;
        boolean alreadyPrefetched;
    }

    private static final class PrefetchBuffers {
        /** 实例级步长状态数组（每个 ImprovedNoise 实例独立跟踪）。 */
        InstanceStepState[] instanceStates = new InstanceStepState[16];
        int instanceCount;

        /** speculativePrefetch 构建预取坐标用（按 5 个 double 为一组：x, y, z, yff, yfv）。 */
        double[] coords = new double[1024 * 5];
        /** prefetchBatch 内部使用的辅助 buffer。 */
        double[] filteredCoords = new double[1024 * 5];
        double[] results = new double[1024];
    }

    private final ThreadLocal<PrefetchBuffers> prefetchBuffers = ThreadLocal.withInitial(PrefetchBuffers::new);

    /**
     * 在 NoiseChunk 初始化时调用，清空当前线程的缓存与预取状态。
     *
     * <p>清空内容：</p>
     * <ul>
     *   <li>{@link NoiseCache}：释放上一个 chunk 的缓存条目，避免跨 chunk 串扰</li>
     *   <li>{@link PrefetchBuffers} 的步长观察状态：上一个 chunk 的 cell 步长
     *       不适用于新 chunk（ImprovedNoise 实例可能不同，或频率参数变化），
     *       必须重置以重新观察。同时释放对上一个 ImprovedNoise 实例的强引用。</li>
     * </ul>
     */
    public void beginChunk() {
        if (Config.verbose) {
            Thread currentThread = Thread.currentThread();
            String threadName = currentThread.getName();
            boolean isGenerationThread = threadName.contains("Chunk Generation")
                    || threadName.contains("chunk-gen")
                    || threadName.contains("ChunkGen")
                    || threadName.startsWith("ForkJoinPool")
                    || threadName.contains("generation")
                    || threadName.contains("Generator");

            if (!isGenerationThread) {
                LOGGER.warn("[nvidia_chunk] [线程诊断] beginChunk 在非生成线程执行：{} (id={})",
                        threadName, currentThread.getId());
            }
        }

        chunksSeen.increment();
        chunkCounter++;

        // 跨 chunk 缓存复用：每 CHUNKS_BEFORE_CLEAR 个 chunk 才清空一次缓存。
        // 相邻 chunk 共享边界噪声值，不清空可提升命中率。
        if (chunkCounter >= CHUNKS_BEFORE_CLEAR) {
            chunkCounter = 0;
            NoiseCache.clearCurrent();
        }

        // 重置实例级步长状态（新 chunk 的 ImprovedNoise 实例不同）
        PrefetchBuffers buffers = prefetchBuffers.get();
        for (int i = 0; i < buffers.instanceCount; i++) {
            buffers.instanceStates[i] = null;
        }
        buffers.instanceCount = 0;

        logStats();
    }

    private void logStats() {
        long total = totalCalls.sum();
        if (total == 0) return;

        long now = System.currentTimeMillis();

        if (!loggedFirstChunk) {
            loggedFirstChunk = true;
            lastStatsLogTime = now;
            LOGGER.info("[nvidia_chunk] ===== 首次区块生成统计 ===== ");
            LOGGER.info("[nvidia_chunk] 总调用: {}, 缓存命中: {}, 命中率: {}%",
                    total, cacheHits.sum(),
                    String.format("%.1f", (double) cacheHits.sum() / total * 100));
            LOGGER.info("[nvidia_chunk] CPU计算: {}, GPU批次数: {}, GPU计算数: {}",
                    cpuComputations.sum(), gpuBatchDispatches.sum(), gpuBatchComputations.sum());
            LOGGER.info("[nvidia_chunk] 预取触发: {}, 预取成功: {}, 预取失败: {}",
                    prefetchTriggers.sum(), prefetchSuccesses.sum(), prefetchFails.sum());
            long prefetchPuts = prefetchPredictedCoords.sum();
            if (prefetchPuts > 0) {
                NoiseCache cache = NoiseCache.get();
                long prefetchHits = cache.getPrefetchHits();
                double prefetchHitRate = (double) prefetchHits / prefetchPuts * 100;
                LOGGER.info("[nvidia_chunk] 预取预测: {}, 预取命中: {}, 预取命中率: {}%",
                        prefetchPuts, prefetchHits, String.format("%.1f", prefetchHitRate));
            }
            if (gpuBatchDispatches.sum() == 0) {
                LOGGER.warn("[nvidia_chunk] 警告：GPU 批次调度为 0！可能原因：");
                LOGGER.warn("[nvidia_chunk]   - GPU 未初始化（检查启动日志中是否有 'GPU 上下文初始化成功'）");
                LOGGER.warn("[nvidia_chunk]   - prefetchSize = 0（配置文件中禁用了预取）");
                LOGGER.warn("[nvidia_chunk]   - GPU 已被禁用（检查日志中是否有 'GPU 调度失败'）");
                LOGGER.warn("[nvidia_chunk]   - 步长观察未稳定（还未观察到足够的坐标差值）");
            } else {
                LOGGER.info("[nvidia_chunk] GPU 已成功调用！显卡 = {} ({})",
                        gpu != null ? gpu.getDeviceName() : "null",
                        gpu != null ? gpu.getVendorName() : "null");
            }
            LOGGER.info("[nvidia_chunk] ===========================");
            return;
        }

        if (now - lastStatsLogTime >= 60000) {
            lastStatsLogTime = now;
            double hitRate = (double) cacheHits.sum() / total * 100;
            double avgPrefetchPerChunk = chunksSeen.sum() > 0
                    ? (double) prefetchTriggers.sum() / chunksSeen.sum() : 0;
            String mode = (Config.PREFETCH_SIZE_INT <= 0 || gpu == null || !gpu.isAvailable() || gpuDisabled)
                    ? "纯缓存（预取已禁用）"
                    : "GPU预取模式";
            LOGGER.info("[nvidia_chunk] 统计（自启动累计，每60秒输出）[{}]: 区块={}, 总调用={}, 命中率={}%, CPU={}, GPU批={}, GPU计算={}, 预取触发={} (平均/chunk={}), 预取成功={}, 预取失败={}",
                    mode, chunksSeen.sum(), total, String.format("%.1f", hitRate),
                    cpuComputations.sum(), gpuBatchDispatches.sum(), gpuBatchComputations.sum(),
                    prefetchTriggers.sum(), String.format("%.2f", avgPrefetchPerChunk),
                    prefetchSuccesses.sum(), prefetchFails.sum());
        }
    }

    private void onGpuFailure(Throwable t) {
        // AtomicInteger：修复旧实现 volatile int ++ 的非原子读-改-写竞争
        int fails = consecutiveFailures.incrementAndGet();
        int threshold = Config.failuresBeforeDisable;
        Config.FallbackStrategy strategy = Config.fallbackStrategy;

        if (strategy == Config.FallbackStrategy.DISABLE) {
            gpuDisabled = true;
            LOGGER.warn("[nvidia_chunk] GPU 调度失败（策略=DISABLE），永久禁用 GPU 路径。原因：{}",
                    t.getMessage());
        } else if (strategy == Config.FallbackStrategy.LOG_ONCE && !loggedOnce) {
            loggedOnce = true;
            LOGGER.warn("[nvidia_chunk] GPU 调度首次失败（策略=LOG_ONCE），后续静默回退 CPU。原因：{}",
                    t.getMessage());
        } else if (Config.verbose) {
            LOGGER.warn("[nvidia_chunk] GPU 调度失败，回退 CPU。原因：{}", t.getMessage());
        }

        if (threshold > 0 && fails >= threshold) {
            gpuDisabled = true;
            LOGGER.warn("[nvidia_chunk] GPU 连续失败 {} 次，达到阈值 {}，自动禁用 GPU 路径",
                    fails, threshold);
        }
    }

    // ------------------------------------------------------------------
    // CPU 回退实现：严格按 MC 1.20.1 ImprovedNoise.noise(DDDDD)D + sampleAndLerp 字节码移植
    // ------------------------------------------------------------------

    /**
     * Java 版 ImprovedNoise.noise，与原版输出逐位一致。
     * 当 GPU 不可用时使用；也用于正确性校验。
     *
     * <p>性能优化（v2）：展平 GRADIENT 表为一维数组，提升缓存局部性；
     * 内联 pIdx/gradDot 的核心计算，减少方法调用开销。JIT 会进一步内联，
     * 但展平数组避免了每次 gradDot 的 int[] 数组寻址。</p>
     */
    public static double cpuNoise(byte[] perm, double xo, double yo, double zo,
                                  double x, double y, double z,
                                  double yFloorFreq, double yFloorValue) {
        double x2 = x + xo;
        double y2 = y + yo;
        double z2 = z + zo;

        int floorX = mcFloor(x2);
        int floorY = mcFloor(y2);
        int floorZ = mcFloor(z2);

        double fracX = x2 - (double) floorX;
        double fracY = y2 - (double) floorY;
        double fracZ = z2 - (double) floorZ;

        double yOffset;
        if (yFloorFreq != 0.0) {
            double d7 = (yFloorValue >= 0.0 && yFloorValue < fracY) ? yFloorValue : fracY;
            yOffset = (double) mcFloor(d7 / yFloorFreq + SHIFT_UP_EPSILON) * yFloorFreq;
        } else {
            yOffset = 0.0;
        }

        return sampleAndLerp(perm, floorX, floorY, floorZ,
                fracX, fracY - yOffset, fracZ, fracY);
    }

    private static double sampleAndLerp(byte[] perm, int floorX, int floorY, int floorZ,
                                        double fracX, double fracYmod, double fracZ,
                                        double fracYorig) {
        // 预计算 floor? & 255，避免在 14 个索引计算中重复 & 255。
        // 数学等价性：(a + floorK) & 255 == (a + (floorK & 255)) & 255，因为 & 255 是 mod 256。
        final int fx  = floorX & 255;
        final int fx1 = (floorX + 1) & 255;
        final int fy  = floorY & 255;
        final int fy1 = (floorY + 1) & 255;
        final int fz  = floorZ & 255;
        final int fz1 = (floorZ + 1) & 255;

        // 内联 pIdx：perm[i & 255] & 255
        int pi  = perm[fx] & 255;
        int pj  = perm[fx1] & 255;
        int pk  = perm[(pi + fy) & 255] & 255;
        int pl  = perm[(pi + fy1) & 255] & 255;
        int pi1 = perm[(pj + fy) & 255] & 255;
        int pj1 = perm[(pj + fy1) & 255] & 255;

        // 内联 gradDot：预计算 g = (h & 15) * 3 作为 GRADIENT_FLAT 基址，
        // 省去每个 gradDot 的 2 次 g*3 乘法（共 16 次乘法 → 8 次乘法）。
        // GRADIENT_FLAT 布局：[g0x, g0y, g0z, g1x, g1y, g1z, ...]，基址 = (hash & 15) * 3。
        int g0 = (perm[(pk + fz) & 255] & 15) * 3;
        double d0 = (double) GRADIENT_FLAT[g0]     * fracX
                  + (double) GRADIENT_FLAT[g0 + 1] * fracYmod
                  + (double) GRADIENT_FLAT[g0 + 2] * fracZ;

        int g1 = (perm[(pi1 + fz) & 255] & 15) * 3;
        double d1 = (double) GRADIENT_FLAT[g1]     * (fracX - 1)
                  + (double) GRADIENT_FLAT[g1 + 1] * fracYmod
                  + (double) GRADIENT_FLAT[g1 + 2] * fracZ;

        int g2 = (perm[(pl + fz) & 255] & 15) * 3;
        double d2 = (double) GRADIENT_FLAT[g2]     * fracX
                  + (double) GRADIENT_FLAT[g2 + 1] * (fracYmod - 1)
                  + (double) GRADIENT_FLAT[g2 + 2] * fracZ;

        int g3 = (perm[(pj1 + fz) & 255] & 15) * 3;
        double d3 = (double) GRADIENT_FLAT[g3]     * (fracX - 1)
                  + (double) GRADIENT_FLAT[g3 + 1] * (fracYmod - 1)
                  + (double) GRADIENT_FLAT[g3 + 2] * fracZ;

        int g4 = (perm[(pk + fz1) & 255] & 15) * 3;
        double d4 = (double) GRADIENT_FLAT[g4]     * fracX
                  + (double) GRADIENT_FLAT[g4 + 1] * fracYmod
                  + (double) GRADIENT_FLAT[g4 + 2] * (fracZ - 1);

        int g5 = (perm[(pi1 + fz1) & 255] & 15) * 3;
        double d5 = (double) GRADIENT_FLAT[g5]     * (fracX - 1)
                  + (double) GRADIENT_FLAT[g5 + 1] * fracYmod
                  + (double) GRADIENT_FLAT[g5 + 2] * (fracZ - 1);

        int g6 = (perm[(pl + fz1) & 255] & 15) * 3;
        double d6 = (double) GRADIENT_FLAT[g6]     * fracX
                  + (double) GRADIENT_FLAT[g6 + 1] * (fracYmod - 1)
                  + (double) GRADIENT_FLAT[g6 + 2] * (fracZ - 1);

        int g7 = (perm[(pj1 + fz1) & 255] & 15) * 3;
        double d7 = (double) GRADIENT_FLAT[g7]     * (fracX - 1)
                  + (double) GRADIENT_FLAT[g7 + 1] * (fracYmod - 1)
                  + (double) GRADIENT_FLAT[g7 + 2] * (fracZ - 1);

        double sx = smoothstep(fracX);
        double sy = smoothstep(fracYorig);
        double sz = smoothstep(fracZ);

        return lerp3(sx, sy, sz, d0, d1, d2, d3, d4, d5, d6, d7);
    }

    private static int mcFloor(double v) {
        int i = (int) v;
        return (v < (double) i) ? i - 1 : i;
    }

    private static double smoothstep(double t) {
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    }

    private static double lerp(double delta, double a, double b) {
        return a + delta * (b - a);
    }

    private static double lerp2(double p1, double p2, double p3, double p4, double p5, double p6) {
        return lerp(p2, lerp(p1, p3, p4), lerp(p1, p5, p6));
    }

    private static double lerp3(double p1, double p2, double p3,
                                double p4, double p5, double p6, double p7,
                                double p8, double p9, double p10, double p11) {
        return lerp(p3,
                lerp2(p1, p2, p4, p5, p6, p7),
                lerp2(p1, p2, p8, p9, p10, p11));
    }

    /**
     * 生成详细统计文本（供 {@code /nvidia_chunk stats} 命令输出与日志共用）。
     */
    public List<String> getDetailedStats() {
        long total = totalCalls.sum();
        long hits = cacheHits.sum();
        long cpu = cpuComputations.sum();
        long gpuBatch = gpuBatchDispatches.sum();
        long gpuCompute = gpuBatchComputations.sum();
        long prefetches = prefetchTriggers.sum();
        long prefetchSuccess = prefetchSuccesses.sum();
        long prefetchFail = prefetchFails.sum();

        double hitRate = total > 0 ? (double) hits / total * 100 : 0;
        double gpuRatio = (cpu + gpuCompute) > 0 ? (double) gpuCompute / (cpu + gpuCompute) * 100 : 0;

        List<String> lines = new ArrayList<>();
        lines.add(String.format("总调用: %d, 缓存命中: %d, 命中率: %.1f%%", total, hits, hitRate));
        lines.add(String.format("CPU计算: %d, GPU计算: %d, GPU占比: %.1f%%", cpu, gpuCompute, gpuRatio));
        lines.add("GPU批次数: " + gpuBatch);
        lines.add(String.format("预取触发: %d, 成功: %d, 失败: %d", prefetches, prefetchSuccess, prefetchFail));

        if (gpuBatch > 0) {
            lines.add(String.format("平均每批: %.1f 次计算", (double) gpuCompute / gpuBatch));
        }
        if (prefetches > 0) {
            lines.add(String.format("预取成功率: %.1f%%", (double) prefetchSuccess / prefetches * 100));
        }

        if (gpuDisabled) {
            lines.add("GPU 已被禁用（连续失败过多）");
        } else if (gpu == null) {
            lines.add("GPU 未初始化");
        } else if (!gpu.isAvailable()) {
            lines.add("GPU 不可用");
        } else {
            lines.add("GPU 状态正常：" + gpu.getDeviceName() + " (" + gpu.getVendorName() + ")");
        }
        return lines;
    }

    public void logDetailedStats() {
        LOGGER.info("[nvidia_chunk] ===== 详细统计 ===== ");
        for (String line : getDetailedStats()) {
            LOGGER.info("[nvidia_chunk] {}", line);
        }
        LOGGER.info("[nvidia_chunk] =====================");
    }
}
