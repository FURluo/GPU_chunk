package furluo.nvidia_chunk.mixin;

import furluo.nvidia_chunk.Config;
import furluo.nvidia_chunk.gpu.GPUNoiseManager;
import net.minecraft.world.level.levelgen.synth.ImprovedNoise;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * ImprovedNoise Mixin：将 {@code noise(double,double,double,double,double)} 调用路由到
 * {@link GPUNoiseManager}，由其决定走 GPU 或 CPU 回退，并使用 {@link furluo.nvidia_chunk.gpu.NoiseCache}
 * 缓存结果。
 *
 * <p>策略：</p>
 * <ul>
 *   <li>{@code Config.enabled == false}：不取消，原版逻辑正常执行（模组完全透明）。</li>
 *   <li>{@code Config.enabled == true}：取消原方法，由管理器返回结果（GPU 或 CPU 回退）。
 *       管理器的 CPU 回退实现严格按原版字节码移植，输出逐位一致。</li>
 *   <li>3 参数版 {@code noise(DDD)D} 内部调用 5 参数版，因此只需 hook 5 参数版即可覆盖全部入口。</li>
 * </ul>
 *
 * <p><b>安全性保证</b>：本 Mixin 在调用 {@link GPUNoiseManager#dispatchNoise} 时包裹
 * try-catch Throwable。任何异常（包括 {@link NoClassDefFoundError}、{@link StackOverflowError}
 * 等）都不会导致原版 {@code noise} 方法异常退出——异常时不调用 {@code cir.setReturnValue()}，
 * 原版逻辑会正常执行。这确保了：</p>
 * <ul>
 *   <li>Jocl 库缺失 / GPU 初始化失败 → 原版逻辑</li>
 *   <li>缓存内部异常 → 原版逻辑</li>
 *   <li>perm 表被其他模组篡改 → 原版逻辑</li>
 *   <li>任何未预期的运行时异常 → 原版逻辑</li>
 * </ul>
 *
 * <p>兼容性：本 Mixin 仅 {@code @Inject} 在 HEAD 处，不修改原方法字节码、不重写字段，
 * 不与任何已知模组冲突（已验证 labs/ 下全部模组未 Mixin ImprovedNoise）。</p>
 */
@Mixin(ImprovedNoise.class)
public abstract class ImprovedNoiseMixin {

    // @Shadow 读取原版私有 / 公有字段，避免使用反射
    @Shadow
    private byte[] p;

    @Shadow
    public double xo;

    @Shadow
    public double yo;

    @Shadow
    public double zo;

    @Inject(method = "noise(DDDDD)D", at = @At("HEAD"), cancellable = true)
    private void nvidia_chunk$onNoise5(double x, double y, double z,
                                       double yFloorFreq, double yFloorValue,
                                       CallbackInfoReturnable<Double> cir) {
        if (!Config.enabled) {
            // 模组禁用：原版逻辑正常执行
            return;
        }
        GPUNoiseManager mgr = GPUNoiseManager.getInstance();
        if (mgr == null) {
            // 管理器未初始化（例如模组加载早期）：原版逻辑正常执行
            return;
        }

        try {
            // 路由到 GPU/CPU 管理器，结果必返回（缓存命中 / GPU / CPU 回退）
            double result = mgr.dispatchNoise(
                    (ImprovedNoise) (Object) this,
                    this.p,
                    this.xo, this.yo, this.zo,
                    x, y, z,
                    yFloorFreq, yFloorValue
            );
            cir.setReturnValue(result);
        } catch (Throwable t) {
            // 关键安全兜底：任何异常都不 setReturnValue，原版逻辑会正常执行。
            // 这保证了即使缓存/GPU/CPU 回退路径全部失败，区块生成也不会崩溃。
            // 日志去重由 GPUNoiseManager.logSuppressedError 负责，避免刷屏。
            GPUNoiseManager.logSuppressedError(t);
        }
    }
}
