package furluo.nvidia_chunk.commands;

import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import com.mojang.brigadier.context.CommandContext;
import com.mojang.brigadier.exceptions.CommandSyntaxException;
import furluo.nvidia_chunk.Config;
import furluo.nvidia_chunk.gpu.GPUContext;
import furluo.nvidia_chunk.gpu.GPUNoiseManager;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.network.chat.Component;
import net.minecraftforge.event.RegisterCommandsEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;

@Mod.EventBusSubscriber(modid = "nvidia_chunk")
public class NvidiaChunkCommand {

    @SubscribeEvent
    public static void onRegisterCommands(RegisterCommandsEvent event) {
        CommandDispatcher<CommandSourceStack> dispatcher = event.getDispatcher();
        LiteralArgumentBuilder<CommandSourceStack> cmd = Commands.literal("nvidia_chunk")
                .requires(source -> source.hasPermission(2))
                .then(Commands.literal("debug").executes(NvidiaChunkCommand::debug))
                .then(Commands.literal("stats").executes(NvidiaChunkCommand::stats))
                .then(Commands.literal("reload").executes(NvidiaChunkCommand::reload));
        dispatcher.register(cmd);
    }

    private static int debug(CommandContext<CommandSourceStack> ctx) throws CommandSyntaxException {
        CommandSourceStack source = ctx.getSource();

        source.sendSuccess(() -> Component.literal("========== nvidia_chunk 调试信息 =========="), false);

        source.sendSuccess(() -> Component.literal("配置:"), false);
        source.sendSuccess(() -> Component.literal("  enabled: " + Config.enabled), false);
        source.sendSuccess(() -> Component.literal("  verbose: " + Config.verbose), false);
        source.sendSuccess(() -> Component.literal("  prefetchSize: " + Config.prefetchSize), false);
        source.sendSuccess(() -> Component.literal("  useGpuForSingleDispatch: " + Config.useGpuForSingleDispatch), false);
        source.sendSuccess(() -> Component.literal("  cacheEnabled: " + Config.cacheEnabled), false);
        source.sendSuccess(() -> Component.literal("  cacheSize: " + Config.cacheSize), false);
        source.sendSuccess(() -> Component.literal("  platformPreference: " + Config.platformPreference), false);
        source.sendSuccess(() -> Component.literal("  deviceIndex: " + Config.deviceIndex), false);
        source.sendSuccess(() -> Component.literal("  workGroupSize: " + Config.workGroupSize), false);

        source.sendSuccess(() -> Component.literal("GPU 状态:"), false);
        GPUContext gpuCtx = GPUContext.getInstance();
        if (gpuCtx == null) {
            source.sendSuccess(() -> Component.literal("  GPUContext: null（未初始化）"), false);
        } else {
            source.sendSuccess(() -> Component.literal("  GPUContext: 已初始化"), false);
            source.sendSuccess(() -> Component.literal("  可用: " + gpuCtx.isAvailable()), false);
            source.sendSuccess(() -> Component.literal("  设备名称: " + gpuCtx.getDeviceName()), false);
            source.sendSuccess(() -> Component.literal("  厂商: " + gpuCtx.getVendorName()), false);
            source.sendSuccess(() -> Component.literal("  工作组大小: " + gpuCtx.getWorkGroupSize()), false);
        }

        source.sendSuccess(() -> Component.literal("管理器状态:"), false);
        GPUNoiseManager mgr = GPUNoiseManager.getInstance();
        if (mgr == null) {
            source.sendSuccess(() -> Component.literal("  GPUNoiseManager: null（未初始化）"), false);
        } else {
            source.sendSuccess(() -> Component.literal("  GPUNoiseManager: 已初始化"), false);
            mgr.logDetailedStats();
        }

        source.sendSuccess(() -> Component.literal("========================================="), false);
        return 1;
    }

    private static int stats(CommandContext<CommandSourceStack> ctx) throws CommandSyntaxException {
        CommandSourceStack source = ctx.getSource();
        GPUNoiseManager mgr = GPUNoiseManager.getInstance();

        source.sendSuccess(() -> Component.literal("========== nvidia_chunk 运行时统计 =========="), false);
        if (mgr != null) {
            mgr.logDetailedStats();
        } else {
            source.sendSuccess(() -> Component.literal("GPUNoiseManager 未初始化"), false);
        }
        source.sendSuccess(() -> Component.literal("==========================================="), false);
        return 1;
    }

    private static int reload(CommandContext<CommandSourceStack> ctx) throws CommandSyntaxException {
        CommandSourceStack source = ctx.getSource();
        source.sendSuccess(() -> Component.literal("[nvidia_chunk] 重新加载配置..."), false);
        Config.reload();
        source.sendSuccess(() -> Component.literal("[nvidia_chunk] 配置已重新加载"), false);
        return 1;
    }
}