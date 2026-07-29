package furluo.nvidia_chunk.mixin;

import com.mojang.logging.LogUtils;
import furluo.nvidia_chunk.Config;
import furluo.nvidia_chunk.gpu.GPUNoiseManager;
import net.minecraft.world.level.levelgen.NoiseChunk;
import org.slf4j.Logger;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * NoiseChunk Mixin：在每个 NoiseChunk 构造完成时清空当前线程的噪声缓存。
 *
 * <p>原版 {@link NoiseChunk} 在 {@code fillSlice} 中通过三角插值计算密度函数，
 * 会在同一区块内对同一 ImprovedNoise 实例的同一坐标多次调用 {@code noise()}。
 * 缓存在一个区块的生命周期内有效，区块切换时清空以释放内存并避免跨区块串扰。</p>
 *
 * <p>线程模型：{@link NoiseChunk} 的构造与 {@code fillSlice} 在同一区块工作线程上执行，
 * {@link furluo.nvidia_chunk.gpu.NoiseCache} 使用 {@link ThreadLocal}，因此清空操作准确作用于当前线程。</p>
 *
 * <p><b>安全性保证</b>：本 Mixin 在调用 {@link GPUNoiseManager#beginChunk()} 时包裹
 * try-catch Throwable。任何异常（缓存内部错误、OOM 等）都不会导致 NoiseChunk
 * 构造失败——异常时不做任何事，原版 NoiseChunk 构造会正常完成。最坏情况下
 * 仅仅是缓存未清空（可能造成轻微跨区块串扰），但绝不会导致区块生成崩溃。</p>
 *
 * <p>兼容性：仅 {@code @Inject} 在构造函数 RETURN，不修改任何逻辑，不与任何已知模组冲突
 * （已验证 labs/ 下全部模组未 Mixin NoiseChunk）。</p>
 */
@Mixin(NoiseChunk.class)
public abstract class NoiseChunkMixin {

    private static final Logger LOGGER = LogUtils.getLogger();

    @Inject(method = "<init>*", at = @At("RETURN"))
    private void nvidia_chunk$onNoiseChunkInit(CallbackInfo ci) {
        try {
            GPUNoiseManager mgr = GPUNoiseManager.getInstance();
            if (mgr != null) {
                mgr.beginChunk();
            }
        } catch (Throwable t) {
            // 安全兜底：清空缓存失败不影响 NoiseChunk 构造。
            // 仅仅是缓存可能未清空（轻微跨区块串扰），但绝不崩溃。
            if (Config.verbose) {
                LOGGER.warn("[nvidia_chunk] beginChunk 清空缓存异常（已忽略）：{}",
                        t.toString());
            }
        }
    }
}
