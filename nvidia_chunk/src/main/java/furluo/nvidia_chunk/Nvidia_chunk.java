package furluo.nvidia_chunk;

import com.mojang.logging.LogUtils;
import furluo.nvidia_chunk.gpu.GPUContext;
import furluo.nvidia_chunk.gpu.GPUNoiseManager;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.event.server.ServerStartingEvent;
import net.minecraftforge.event.server.ServerStoppingEvent;
import net.minecraftforge.eventbus.api.IEventBus;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.DistExecutor;
import net.minecraftforge.fml.ModLoadingContext;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.config.ModConfig;
import net.minecraftforge.fml.event.lifecycle.FMLCommonSetupEvent;
import net.minecraftforge.fml.javafmlmod.FMLJavaModLoadingContext;
import net.minecraftforge.fml.loading.FMLEnvironment;
import org.slf4j.Logger;

/**
 * nvidia_chunk 模组主类（纯服务端 / 集成服务端）。
 *
 * <p>使用 OpenCL 调用 NVIDIA / AMD / Intel 显卡并行加速 Minecraft 区块噪声生成，
 * 同时保留原版世界生成的全部特性，兼容修改地形生成的模组。</p>
 *
 * <p><b>服务端专属设计</b>：</p>
 * <ul>
 *   <li>本模组只对<b>服务端（含集成服务端）</b>有意义。客户端单独安装无任何效果，
 *       但因 mods.toml 设置了 <code>displayTest = IGNORE_ALL_VERSION</code>，客户端可自由进出。</li>
 *   <li>Mixins 注入 ImprovedNoise.noise() 和 NoiseChunk 构造，这两个类在集成服务端
 *       （Minecraft 客户端内置服务器）也会被加载，故 Mixin 必须声明在 <code>mixins</code>
 *       而非 <code>client</code>。</li>
 *   <li>GPU 初始化在 {@link FMLCommonSetupEvent}，通过 {@link DistExecutor#safeRunForDist}
 *       在不同物理端上分别执行。</li>
 * </ul>
 *
 * <p>启动流程：</p>
 * <ol>
 *   <li>构造函数：注册配置（{@link Config#SPEC}）</li>
 *   <li>{@link FMLCommonSetupEvent}：按物理端分流，物理服务端初始化 {@link GPUContext}</li>
 *   <li>初始化 {@link GPUNoiseManager}（管理器单例就绪后，Mixin 即开始将 noise() 路由到 GPU）</li>
 *   <li>{@link ServerStoppingEvent}：释放 GPU 资源</li>
 * </ol>
 *
 * <p>若 GPU 不可用（无 OpenCL 设备 / 不支持双精度 / 内核编译失败），
 * 管理器自动回退到 CPU 实现，仍享受线程本地缓存加速。</p>
 */
@Mod(Nvidia_chunk.MODID)
public class Nvidia_chunk {

    public static final String MODID = "nvidia_chunk";
    private static final Logger LOGGER = LogUtils.getLogger();

    public Nvidia_chunk() {
        IEventBus modEventBus = FMLJavaModLoadingContext.get().getModEventBus();
        ModLoadingContext.get().registerConfig(ModConfig.Type.COMMON, Config.SPEC);
        MinecraftForge.EVENT_BUS.register(this);
        modEventBus.addListener(this::commonSetup);

        // 打印模组物理端上下文，便于排错
        Dist dist = FMLEnvironment.dist;
        LOGGER.info("[nvidia_chunk] 模组加载，物理端 = {} ({})",
                dist, dist.isDedicatedServer() ? "专用服务器" : (dist.isClient() ? "客户端/集成服务端" : "?"));
    }

    /**
     * 在 FMLCommonSetupEvent 初始化 GPU。此时配置已加载，可以读取 {@link Config}。
     *
     * <p>使用 enqueueWork 确保初始化在主线程执行，避免 OpenCL 原生库加载的线程安全问题。</p>
     * <p>通过 DistExecutor 按物理端分流：客户端物理端不分配 GPU（仅打印 INFO）。</p>
     */
    private void commonSetup(final FMLCommonSetupEvent event) {
        event.enqueueWork(() -> {
            // 物理端分流：DistExecutor 在打包时按物理端选边
            // 客户端物理端 -> 不分配 OpenCL 资源
            // 服务端/集成服务端 -> 完整初始化
            DistExecutor.safeRunForDist(
                    () -> () -> { /* 客户端物理端不做事 */ return Boolean.FALSE; },
                    () -> () -> { initializeGpu(); return Boolean.TRUE; }
            );
        });
    }

    private void initializeGpu() {
        // 进入此方法时已确认是服务端物理端（专用服务端或集成服务端）。
        // 客户端物理端已在 commonSetup 中通过 DistExecutor 跳过此方法。

        if (!Config.enabled) {
            LOGGER.info("[nvidia_chunk] GPU 加速已在配置中禁用，模组将以透明模式运行（不影响原版逻辑）");
            GPUNoiseManager.init(null);
            return;
        }

        // 先检查 JOCL 库是否可用
        try {
            Class.forName("org.jocl.CL");
        } catch (Throwable e) {
            LOGGER.warn("[nvidia_chunk] 未找到 JOCL 库（org.jocl.CL），GPU 加速不可用。");
            LOGGER.warn("[nvidia_chunk] 请确保服务器中安装了包含 JOCL 的模组（如 quantified api-forge），");
            LOGGER.warn("[nvidia_chunk] 或将 jocl-x.x.x.jar 放入服务器的 libraries 目录。");
            LOGGER.warn("[nvidia_chunk] 将使用 CPU 回退 + 缓存，区块生成仍会加速（得益于缓存）。");
            GPUNoiseManager.init(null);
            return;
        }

        LOGGER.info("[nvidia_chunk] 开始初始化 GPU（首选厂商 = {}，设备序号 = {}，工作组大小 = {}）",
                Config.platformPreference, Config.deviceIndex, Config.workGroupSize);

        GPUContext gpu = GPUContext.init(Config.platformPreference, Config.deviceIndex, Config.workGroupSize);
        if (gpu == null || !gpu.isAvailable()) {
            LOGGER.warn("[nvidia_chunk] GPU 不可用，将使用 CPU 回退 + 缓存。区块生成仍会加速（得益于缓存）。");
        }

        GPUNoiseManager.init(gpu);

        Runtime.getRuntime().addShutdownHook(new Thread(this::shutdownGpu, "nvidia_chunk-shutdown"));
    }

    /**
     * 服务器启动时确认运行模式（仅在专用服务端或集成服务端运行）。
     */
    @SubscribeEvent
    public void onServerStarting(ServerStartingEvent event) {
        LOGGER.info("[nvidia_chunk] ServerStartingEvent - 服务端启用 GPU 区块生成加速");
    }

    @SubscribeEvent
    public void onServerStopping(ServerStoppingEvent event) {
        shutdownGpu();
    }

    private void shutdownGpu() {
        try {
            GPUContext ctx = GPUContext.getInstance();
            if (ctx != null) {
                ctx.close();
            }
        } catch (Throwable t) {
            LOGGER.warn("[nvidia_chunk] GPU 关闭异常：{}", t.getMessage());
        }
    }
}
