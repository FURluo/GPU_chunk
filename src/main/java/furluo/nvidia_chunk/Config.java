package furluo.nvidia_chunk;

import net.minecraftforge.common.ForgeConfigSpec;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.event.config.ModConfigEvent;

/**
 * GPU 噪声加速配置。
 *
 * <p>大部分运行时参数（缓存开关、预取阈值等）可通过 {@code /nvidia_chunk reload}
 * 即时重载生效。但 GPU 上下文相关项（{@code platformPreference} / {@code deviceIndex} /
 * {@code workGroupSize}）仅在 GPU 初始化时读取，<b>需重启游戏/服务器才能生效</b>
 * （GPU 上下文不支持运行时重建）。</p>
 */
@Mod.EventBusSubscriber(modid = Nvidia_chunk.MODID, bus = Mod.EventBusSubscriber.Bus.MOD)
public class Config {

    private static final ForgeConfigSpec.Builder BUILDER = new ForgeConfigSpec.Builder();

    // ------------------------------------------------------------------
    // 基础开关
    // ------------------------------------------------------------------
    private static final ForgeConfigSpec.BooleanValue ENABLED = BUILDER
            .comment("是否启用 GPU 噪声加速。关闭后所有计算回退到原版 CPU 实现，不破坏任何世界生成。")
            .define("enabled", true);

    private static final ForgeConfigSpec.BooleanValue VERBOSE = BUILDER
            .comment("是否在日志中输出 GPU 初始化、批次调度、回退等详细信息（建议仅排错时开启）。")
            .define("verbose", false);

    // ------------------------------------------------------------------
    // 平台 / 设备选择
    // ------------------------------------------------------------------
    private static final ForgeConfigSpec.ConfigValue<String> PLATFORM_PREFERENCE = BUILDER
            .comment("首选 GPU 厂商：auto / nvidia / amd / intel。",
                    "auto 时按 NVIDIA > AMD > Intel 顺序自动选择首个可用 GPU。",
                    "若指定厂商不存在则回退 auto。")
            .define("platformPreference", "auto");

    private static final ForgeConfigSpec.IntValue DEVICE_INDEX = BUILDER
            .comment("在选定厂商平台上的设备序号（0 表示首个 GPU，1 表示第二个，依此类推）。")
            .defineInRange("deviceIndex", 0, 0, 31);

    // ------------------------------------------------------------------
    // OpenCL 调度参数
    // ------------------------------------------------------------------
    private static final ForgeConfigSpec.IntValue WORK_GROUP_SIZE = BUILDER
            .comment("OpenCL 工作组大小，必须为 32 的倍数且不超过设备上限。",
                    "NVIDIA 通常 32 / 64 / 128，AMD 建议 64，Intel 核显建议 16 或 32。")
            .defineInRange("workGroupSize", 64, 1, 1024);

    private static final ForgeConfigSpec.IntValue BATCH_THRESHOLD = BUILDER
            .comment("当单个 NoiseChunk.fillSlice 调用中累计的同 ImprovedNoise 实例 noise() 调用",
                    "次数达到该阈值时触发一次 GPU 批次提交；过小会增加调度开销，过大延迟回填。",
                    "设为 1 等价于每次调用都同步提交（最保守，用于验证正确性）。")
            .defineInRange("batchThreshold", 1024, 1, 1 << 20);

    private static final ForgeConfigSpec.IntValue MAX_BATCH_SIZE = BUILDER
            .comment("单个 GPU 批次最大输入项数（防止内存爆炸），到上限后强制提交。")
            .defineInRange("maxBatchSize", 1 << 16, 256, 1 << 22);

    private static final ForgeConfigSpec.BooleanValue USE_GPU_FOR_SINGLE_DISPATCH = BUILDER
            .comment("是否在缓存未命中时使用 GPU 单次调度（而非 CPU 回退）。",
                    "默认 false：GPU 单次调度开销（~50-200μs 内核启动 + 同步读回）远高于 CPU 计算（~100ns），",
                    "因此默认走 CPU + 缓存，速度更快。仅用于正确性验证或特殊场景才设为 true。",
                    "真正的 GPU 加速通过批量预取（prefetchSize）实现。")
            .define("useGpuForSingleDispatch", false);

    private static final ForgeConfigSpec.IntValue PREFETCH_SIZE = BUILDER
            .comment("GPU 预取的坐标数量（每实例每缓存周期一次）。",
                    "0 = 禁用 GPU 预取（纯 CPU + 缓存模式）。",
                    "推荐值：256~512。默认 256。",
                    "  每个 ImprovedNoise 实例独立观察步长，稳定后一次性预测 256 个坐标，",
                    "  批量提交 GPU 计算（~200μs），预填缓存。",
                    "  每实例每 4 个 chunk 仅触发一次，调度开销可控。",
                    "最大值：512。")
            .defineInRange("prefetchSize", 256, 0, 512);

    // ------------------------------------------------------------------
    // 缓存
    // ------------------------------------------------------------------
    private static final ForgeConfigSpec.BooleanValue CACHE_ENABLED = BUILDER
            .comment("是否启用线程本地噪声结果 LRU 缓存。原版 NoiseChunk 在三角插值时会对同一坐标",
                    "多次调用同一 ImprovedNoise，缓存可消除 GPU 重复计算。强烈建议开启。")
            .define("cacheEnabled", true);

    private static final ForgeConfigSpec.IntValue CACHE_SIZE = BUILDER
            .comment("每个区块工作线程的 LRU 缓存容量（条目数）。建议 4096~65536。")
            .defineInRange("cacheSize", 1 << 14, 256, 1 << 20);

    // ------------------------------------------------------------------
    // 回退策略
    // ------------------------------------------------------------------
    private static final ForgeConfigSpec.EnumValue<FallbackStrategy> FALLBACK = BUILDER
            .comment("GPU 失败时的回退策略：",
                    "SILENT  - 静默回退到 CPU，继续运行（推荐）。",
                    "LOG_ONCE - 首次失败时记录 WARN，之后静默。",
                    "DISABLE  - 永久禁用 GPU 路径直到下次手动重载。")
            .defineEnum("fallbackStrategy", FallbackStrategy.SILENT);

    private static final ForgeConfigSpec.IntValue FAILURES_BEFORE_DISABLE = BUILDER
            .comment("当 fallbackStrategy != DISABLE 时，连续失败达到该次数后强制禁用 GPU 路径。",
                    "0 表示永不自动禁用。")
            .defineInRange("failuresBeforeDisable", 16, 0, 100000);

    // ------------------------------------------------------------------
    // 兼容性
    // ------------------------------------------------------------------
    private static final ForgeConfigSpec.BooleanValue RESPECT_MOD_NOISE = BUILDER
            .comment("是否对反射检测到的非原版 ImprovedNoise 子类走 CPU 路径。",
                    "开启时若某 ImprovedNoise 实例的 permutation 表与原版初始化结果长度不符，",
                    "或字段被外部模组修改，则该实例回退 CPU。强烈建议开启以保证兼容性。")
            .define("respectModNoise", true);

    static final ForgeConfigSpec SPEC = BUILDER.build();

    // ------------------------------------------------------------------
    // 运行时缓存字段
    // ------------------------------------------------------------------
    public static boolean enabled;
    public static boolean verbose;
    public static String platformPreference;
    public static int deviceIndex;
    public static int workGroupSize;
    public static int batchThreshold;
    public static int maxBatchSize;
    public static boolean useGpuForSingleDispatch;
    /** 预取是否启用（= PREFETCH_SIZE_INT > 0）。 */
    public static boolean prefetchEnabled;
    public static int PREFETCH_SIZE_INT;  // 原始 int 值，供预取使用
    public static boolean cacheEnabled;
    public static int cacheSize;
    public static FallbackStrategy fallbackStrategy;
    public static int failuresBeforeDisable;
    public static boolean respectModNoise;

    /**
     * GPU 失败时的回退策略。
     */
    public enum FallbackStrategy {
        /** 静默回退到 CPU，继续尝试 GPU（直到 failuresBeforeDisable）。 */
        SILENT,
        /** 首次失败记录 WARN，之后静默。 */
        LOG_ONCE,
        /** 立即永久禁用 GPU 路径。 */
        DISABLE
    }

    @SubscribeEvent
    static void onLoad(final ModConfigEvent event) {
        reload();
    }

    /**
     * 运行时判断是否应使用 GPU 路径。Mixins 在每次调用前检查。
     */
    public static boolean useGpu() {
        return enabled;
    }

    /**
     * 重新读取所有配置值。注意：platformPreference / deviceIndex / workGroupSize
     * 仅在 GPU 初始化时读取，重载后不会重建 GPU 上下文（需重启）。
     */
    public static void reload() {
        enabled = ENABLED.get();
        verbose = VERBOSE.get();
        platformPreference = PLATFORM_PREFERENCE.get().toLowerCase();
        deviceIndex = DEVICE_INDEX.get();
        workGroupSize = WORK_GROUP_SIZE.get();
        batchThreshold = BATCH_THRESHOLD.get();
        maxBatchSize = MAX_BATCH_SIZE.get();
        useGpuForSingleDispatch = USE_GPU_FOR_SINGLE_DISPATCH.get();
        PREFETCH_SIZE_INT = PREFETCH_SIZE.get();
        prefetchEnabled = PREFETCH_SIZE_INT > 0;
        cacheEnabled = CACHE_ENABLED.get();
        cacheSize = CACHE_SIZE.get();
        fallbackStrategy = FALLBACK.get();
        failuresBeforeDisable = FAILURES_BEFORE_DISABLE.get();
        respectModNoise = RESPECT_MOD_NOISE.get();
    }
}
