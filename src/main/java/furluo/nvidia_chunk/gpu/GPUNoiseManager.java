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

        // 线程诊断：检测噪声计算的线程
        if (Config.verbose && totalCalls.sum() <= 5) {
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
                observeStep(instance, x, y, z, buffers);
                if (buffers.stepStable) {
                    boolean fire;
                    if (!buffers.alreadyPrefetched) {
                        buffers.alreadyPrefetched = true;
                        buffers.missCountSinceLastPrefetch = 0L;
                        fire = true;
                    } else {
                        buffers.missCountSinceLastPrefetch++;
                        fire = buffers.missCountSinceLastPrefetch >= Config.PREFETCH_SIZE_INT;
                        if (fire) {
                            buffers.missCountSinceLastPrefetch = 0L;
                        }
                    }
                    if (fire) {
                        prefetchTriggers.increment();
                        speculativePrefetch(instance, perm, xo, yo, zo,
                                x, y, z, yFloorFreq, yFloorValue, buffers);
                    }
                }
            } catch (Throwable t) {
                // 预取异常：完全静默，不影响主路径
                prefetchFails.increment();
                if (Config.verbose) {
                    LOGGER.warn("[nvidia_chunk] 预取异常（已忽略）：{}", t.toString());
                }
            }
        }

        return result;
    }

    /**
     * 观察连续未命中坐标的差值，自适应推测 cell 三轴步长。
     *
     * <p>原理：{@code NoiseChunk.fillSlice} 对每个 cell 的 8 个顶点调用
     * {@code ImprovedNoise.noise}，8 个顶点的 (x, y, z) 坐标构成
     * (x0 + i*dx, y0 + j*dy, z0 + k*dz) 的 8 种组合（i,j,k ∈ {0,1}），
     * 其中 dx = cellWidth * xFrequency，dy = cellHeight * yFrequency，
     * dz = cellWidth * zFrequency。</p>
     *
     * <p>因此连续两次未命中的差值必然是 (±dx, 0, 0)、(0, ±dy, 0)、(0, 0, ±dz)
     * 或其组合。我们记录每个轴出现过的最大绝对差值作为该轴的步长。
     * 当三轴步长都已观察到且观察次数足够时，标记步长稳定。</p>
     *
     * <p>稳定性判断：观察次数 >= 4（覆盖 cell 8 顶点中的至少 4 个不同方向），
     * 且三轴步长都非零。步长变化时（不同绝对值的差值出现）重置稳定标记。</p>
     */
    private void observeStep(ImprovedNoise instance, double x, double y, double z,
                             PrefetchBuffers buffers) {
        if (buffers.lastInstance != instance) {
            if (Config.verbose) {
                LOGGER.info("[nvidia_chunk] [调试] 新 ImprovedNoise 实例，重置步长观察");
            }
            buffers.lastInstance = instance;
            buffers.lastX = x;
            buffers.lastY = y;
            buffers.lastZ = z;
            buffers.stepX = 0;
            buffers.stepY = 0;
            buffers.stepZ = 0;
            buffers.observations = 1;
            buffers.stepStable = false;
            buffers.alreadyPrefetched = false;
            return;
        }

        double dx = x - buffers.lastX;
        double dy = y - buffers.lastY;
        double dz = z - buffers.lastZ;

        // 更新上次坐标（无论差值是否为零都更新，反映最新位置）
        buffers.lastX = x;
        buffers.lastY = y;
        buffers.lastZ = z;
        buffers.observations++;

        // 同一坐标重复调用（差值全零）：不更新步长，但也不重置
        if (dx == 0 && dy == 0 && dz == 0) {
            if (Config.verbose) {
                LOGGER.info("[nvidia_chunk] [调试] 坐标重复, 观察次数={}, 步长=({},{},{})",
                        buffers.observations, buffers.stepX, buffers.stepY, buffers.stepZ);
            }
            return;
        }

        // 更新各轴步长：取出现过的最大绝对差值
        // 注意：Minecraft 区块生成中，X 和 Z 是单元格边界（固定步长），
        // 而 Y 是地形高度（随机变化），所以只观察 X 和 Z 步长。
        boolean stepChanged = false;
        if (dx != 0) {
            if (buffers.stepX == 0) {
                buffers.stepX = dx;
                if (Config.verbose) {
                    LOGGER.info("[nvidia_chunk] [调试] 观察到 X 轴步长: {}", dx);
                }
            } else {
                double absDx = Math.abs(dx);
                double absStep = Math.abs(buffers.stepX);
                if (absDx > absStep) {
                    if (Config.verbose) {
                        LOGGER.info("[nvidia_chunk] [调试] X 轴步长更新: {} -> {}", buffers.stepX, dx);
                    }
                    buffers.stepX = dx;
                    stepChanged = true;
                }
            }
        }
        // Y 轴不参与步长观察（Y 是地形高度，不是固定步长）
        if (dz != 0) {
            if (buffers.stepZ == 0) {
                buffers.stepZ = dz;
                if (Config.verbose) {
                    LOGGER.info("[nvidia_chunk] [调试] 观察到 Z 轴步长: {}", dz);
                }
            } else {
                double absDz = Math.abs(dz);
                double absStep = Math.abs(buffers.stepZ);
                if (absDz > absStep) {
                    if (Config.verbose) {
                        LOGGER.info("[nvidia_chunk] [调试] Z 轴步长更新: {} -> {}", buffers.stepZ, dz);
                    }
                    buffers.stepZ = dz;
                    stepChanged = true;
                } else if (absDz < absStep && absDz > 0) {
                    if (Config.verbose) {
                        LOGGER.info("[nvidia_chunk] [调试] Z 轴出现亚步长: {} (主步长={})", dz, buffers.stepZ);
                    }
                }
            }
        }

        // 步长变化时重置稳定标记，需要重新观察
        // 注意：一旦步长稳定，不再因为后续步长变化而重置，避免循环重置导致预取无法触发
        // 预取基于当前观察到的步长进行，即使后续步长变化，预取仍然有效（只是命中率可能降低）
        if (stepChanged && !buffers.stepStable) {
            buffers.alreadyPrefetched = false;
            if (Config.verbose) {
                LOGGER.info("[nvidia_chunk] [调试] 步长变化（尚未稳定），重置预取状态");
            }
        }

        // X 和 Z 步长都已观察到（Y 不参与），且观察次数足够（>=4），标记稳定
        boolean wasUnstable = !buffers.stepStable;
        if (wasUnstable
                && buffers.stepX != 0
                && buffers.stepZ != 0
                && buffers.observations >= 4) {
            buffers.stepStable = true;
            if (Config.verbose) {
                LOGGER.info("[nvidia_chunk] [调试] 步长观察稳定！观察次数={}, X步长={}, Z步长={}",
                        buffers.observations, buffers.stepX, buffers.stepZ);
            }
        }

        if (Config.verbose) {
            LOGGER.info("[nvidia_chunk] [调试] 观察状态: 次数={}, X步长={}, Z步长={}, 稳定={}",
                    buffers.observations, buffers.stepX, buffers.stepZ, buffers.stepStable);
        }
    }

    /**
     * 自适应批量预取：用观察到的 cell 步长预测相邻坐标，批量提交 GPU 预填缓存。
     *
     * <p>预测策略（按优先级）：</p>
     * <ol>
     *   <li><b>当前 cell 的 7 个其他顶点</b>：基于当前 (x, y, z) 和推测的
     *       (stepX, stepY, stepZ)，生成 (x + i*stepX, y + j*stepY, z + k*stepZ)
     *       的 7 种组合（i,j,k ∈ {0,1}，排除 (0,0,0)）。
     *       这 7 个顶点会在当前 cell 的后续 noise 调用中命中。</li>
     *   <li><b>沿主轴预测相邻 cell</b>：剩余配额沿绝对值最大的轴递增预测，
     *       例如主轴为 x 时预测 (x + n*stepX, y, z)。相邻 cell 的第一个顶点
     *       与当前 cell 的第二个顶点共享，命中率较高。</li>
     * </ol>
     *
     * <p>预取的坐标会通过 {@link #prefetchBatch} 筛选（跳过已命中的），
     * 批量提交 GPU 计算后写入缓存。</p>
     *
     * <p>注意：本方法假设 {@link PrefetchBuffers#stepStable} 已为 true，
     * 调用前由 {@link #dispatchNoise} 判断。</p>
     */
    private void speculativePrefetch(ImprovedNoise instance, byte[] perm,
                                     double xo, double yo, double zo,
                                     double x, double y, double z,
                                     double yFloorFreq, double yFloorValue,
                                     PrefetchBuffers buffers) {
        int n = Config.PREFETCH_SIZE_INT;
        if (n <= 0) return;

        // 运行时安全检查：强制限制预取大小，防止配置过大导致服务器卡顿
        if (n > MAX_SAFE_PREFETCH) {
            if (Config.verbose) {
                LOGGER.warn("[nvidia_chunk] 预取大小 {} 超过安全上限 {}，已截断", n, MAX_SAFE_PREFETCH);
            }
            n = MAX_SAFE_PREFETCH;
        }

        if (buffers.coords.length < n * 5) {
            buffers.coords = new double[Math.max(n * 5, buffers.coords.length * 2)];
        }

        double sx = buffers.stepX;
        double sz = buffers.stepZ;

        int idx = 0;

        // 1) 预测当前 cell 的 3 个其他 XZ 平面顶点
        // 只在 XZ 平面预测（不预测 Y，因为 Y 是地形高度，不是固定步长）
        // 4 顶点 = (i,k) ∈ {0,1}^2 的组合，排除 (0,0)（当前顶点）
        for (int i = 0; i < 2 && idx < n; i++) {
            for (int k = 0; k < 2 && idx < n; k++) {
                if (i == 0 && k == 0) continue;
                int off = idx * 5;
                buffers.coords[off]     = x + i * sx;
                buffers.coords[off + 1] = y;  // Y 坐标保持不变
                buffers.coords[off + 2] = z + k * sz;
                buffers.coords[off + 3] = yFloorFreq;
                buffers.coords[off + 4] = yFloorValue;
                idx++;
            }
        }

        // 2) 剩余配额：沿主轴（X 或 Z）递增预测相邻 cell
        double absSx = Math.abs(sx);
        double absSz = Math.abs(sz);

        // 选择主轴
        double mainStep;
        int mainAxis;  // 0=x, 2=z
        if (absSx >= absSz) {
            mainStep = sx;
            mainAxis = 0;
        } else {
            mainStep = sz;
            mainAxis = 2;
        }

        while (idx < n) {
            int step = idx + 1;  // 从 1 开始，避免与当前顶点重复
            int off = idx * 5;
            if (mainAxis == 0) {
                buffers.coords[off]     = x + step * mainStep;
                buffers.coords[off + 1] = y;
                buffers.coords[off + 2] = z;
            } else {
                buffers.coords[off]     = x;
                buffers.coords[off + 1] = y;
                buffers.coords[off + 2] = z + step * mainStep;
            }
            buffers.coords[off + 3] = yFloorFreq;
            buffers.coords[off + 4] = yFloorValue;
            idx++;
        }

        if (idx == 0) return;

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
     * 线程本地的预取缓冲区（避免每次预取分配数组）。
     *
     * <p>包含两组坐标数组以避免循环依赖：</p>
     * <ul>
     *   <li>{@code coords} - {@link #speculativePrefetch} 构建预取坐标用</li>
     *   <li>{@code filteredCoords} - {@link #prefetchBatch} 筛选未命中项后存储用
     *       （不能与输入 coords 是同一数组，否则 System.arraycopy 会冲突）</li>
     * </ul>
     *
     * <p>步长观察字段用于自适应预取：</p>
     * <ul>
     *   <li>{@code lastInstance}/{@code lastX/Y/Z} - 上次未命中的坐标，用于计算差值</li>
     *   <li>{@code stepX/Y/Z} - 推测的 cell 三轴步长（cellWidth * frequency）</li>
     *   <li>{@code observations} - 观察次数，达到阈值后判定步长稳定</li>
     *   <li>{@code stepStable} - 步长是否已稳定（三轴都非零且观察次数足够）</li>
     *   <li>{@code alreadyPrefetched} - 步长稳定后是否已触发首次预取</li>
     * </ul>
     */
    private static final class PrefetchBuffers {
        /** speculativePrefetch 构建预取坐标用（按 5 个 double 为一组：x, y, z, yff, yfv）。 */
        double[] coords = new double[1024 * 5];
        /** 自上次预取以来的未命中次数（用于节流）。 */
        long missCountSinceLastPrefetch;
        /** 步长稳定后是否已触发首次预取。 */
        boolean alreadyPrefetched;

        // prefetchBatch 内部使用的辅助 buffer（避免与 input 同数组冲突）
        double[] filteredCoords = new double[1024 * 5];
        double[] results = new double[1024];

        // 步长观察字段
        ImprovedNoise lastInstance;
        double lastX, lastY, lastZ;
        double stepX, stepY, stepZ;
        int observations;
        boolean stepStable;
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
        // 线程诊断：检测区块生成线程
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

        NoiseCache.clearCurrent();

        PrefetchBuffers buffers = prefetchBuffers.get();
        buffers.lastInstance = null;
        buffers.lastX = 0;
        buffers.lastY = 0;
        buffers.lastZ = 0;
        buffers.stepX = 0;
        buffers.stepZ = 0;
        buffers.observations = 0;
        buffers.stepStable = false;
        buffers.alreadyPrefetched = false;
        buffers.missCountSinceLastPrefetch = 0L;
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
