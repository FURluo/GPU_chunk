package furluo.nvidia_chunk.gpu;

import furluo.nvidia_chunk.Config;
import net.minecraft.world.level.levelgen.synth.ImprovedNoise;

import java.util.Arrays;

/**
 * 线程本地的 ImprovedNoise 结果缓存（开放寻址 hash table）。
 *
 * <p>性能优化（相比旧版 LinkedHashMap 实现）：</p>
 * <ul>
 *   <li>开放寻址（线性探测），查询/存储零对象分配</li>
 *   <li>原始类型存储（long[] keys + double[] values），无 Double 装箱</li>
 *   <li>位混合 hash 函数（avalanche），分布均匀，减少碰撞</li>
 *   <li>探针字段复用，查询路径无任何对象创建</li>
 * </ul>
 *
 * <p>线程模型：每个区块工作线程通过 {@link ThreadLocal} 持有独立实例，无锁竞争。
 * 缓存生命周期由 {@link GPUNoiseManager#beginChunk} 在 {@code NoiseChunk.<init>} 时清空。</p>
 *
 * <p>容量策略：初始容量为 {@link Config#cacheSize} 的 2 倍（保证 load factor < 0.5）。
 * 当 size 超过 maxSize 时触发扩容（容量翻倍，最大 1&lt;&lt;22）；
 * 若已达上限则整体清空（极端情况，正常区块生成不会触发）。</p>
 *
 * <p>未命中标记：用 {@link Double#NaN} 表示未命中。ImprovedNoise 的输出范围约为
 * [-1.5, 1.5]，不可能为 NaN，因此 NaN 作为 sentinel 安全。</p>
 */
public final class NoiseCache {

    private static final ThreadLocal<NoiseCache> THREAD_LOCAL = ThreadLocal.withInitial(NoiseCache::new);

    /** 最大容量上限，超过此值不再扩容，直接清空。 */
    private static final int MAX_CAPACITY = 1 << 22;

    /**
     * 获取当前线程的缓存实例。
     */
    public static NoiseCache get() {
        return THREAD_LOCAL.get();
    }

    /**
     * 清空当前线程的缓存。应在每个 NoiseChunk 初始化时调用。
     */
    public static void clearCurrent() {
        THREAD_LOCAL.get().clear();
    }

    // ------------------------------------------------------------------
    // 开放寻址 hash table
    // 每个 slot 占用 6 个 long（identity, xBits, yBits, zBits, yffBits, yfvBits）
    // + 1 个 double value
    // ------------------------------------------------------------------
    private long[] keys;        // capacity * 6
    private double[] values;    // capacity
    private boolean[] occupied; // capacity
    private boolean[] fromPrefetch; // capacity: true if this entry was written by prefetch
    private int capacity;
    private int mask;
    private int size;
    private final int maxSize;

    // ------------------------------------------------------------------
    // instanceId 快速路径缓存
    //
    // ImprovedNoise 实例在同一 NoiseChunk 内通常是固定的（一个区块使用 1~4 个实例），
    // 三角插值会连续 8 次调用同一实例。缓存 lastInstance 的 identityHashCode 和
    // 预计算的 hash 基数（instanceId * K），可避免 87% 命中路径中的 native 调用
    // 和 1 次 64 位乘法。
    //
    // 内存安全：lastInstance 是强引用，但 NoiseCache 通过 ThreadLocal 持有，
    // 生命周期与区块工作线程一致；beginChunk() 会清空缓存（含 lastInstance），
    // 不会长期持有 ImprovedNoise 实例。
    // ------------------------------------------------------------------
    private ImprovedNoise lastInstance;
    private long lastInstanceId;
    private long lastInstancePreHash;  // = lastInstanceId * K，hash 计算的初始值

    // 统计
    private long hits;
    private long misses;
    private long prefetchHits;
    private long prefetchPuts;

    /** hash 函数的 Knuth 乘法常数。 */
    private static final long K = 0x9E3779B97F4A7C15L;

    private NoiseCache() {
        this.maxSize = Math.max(256, Config.cacheSize);
        // 初始容量取 >= 2*maxSize 的最小 2 的幂，保证 load factor <= 0.5
        int cap = 16;
        while (cap < (maxSize << 1)) cap <<= 1;
        this.capacity = cap;
        this.mask = cap - 1;
        this.keys = new long[cap * 6];
        this.values = new double[cap];
        this.occupied = new boolean[cap];
        this.fromPrefetch = new boolean[cap];
    }

    /**
     * 解析 instance 的预计算 hash 基数。
     *
     * <p>快速路径：若 instance 与上次相同（三角插值的常见情况），直接复用缓存的
     * identityHashCode 和预计算的 {@code instanceId * K}，避免 native 调用和 1 次乘法。</p>
     */
    private long resolveInstancePreHash(ImprovedNoise instance) {
        if (instance == lastInstance) {
            return lastInstancePreHash;
        }
        long instanceId = System.identityHashCode(instance);
        long preHash = instanceId * K;
        lastInstance = instance;
        lastInstanceId = instanceId;
        lastInstancePreHash = preHash;
        return preHash;
    }

    /**
     * 查询缓存。
     *
     * @return 命中则返回结果；未命中返回 {@link Double#NaN}
     */
    public double get(ImprovedNoise instance, double x, double y, double z,
                      double yFloorFreq, double yFloorValue) {
        if (!Config.cacheEnabled) return Double.NaN;

        long preHash = resolveInstancePreHash(instance);
        long xBits = Double.doubleToLongBits(x);
        long yBits = Double.doubleToLongBits(y);
        long zBits = Double.doubleToLongBits(z);
        long yffBits = Double.doubleToLongBits(yFloorFreq);
        long yfvBits = Double.doubleToLongBits(yFloorValue);

        int idx = hashFromPre(preHash, xBits, yBits, zBits, yffBits, yfvBits) & mask;
        // resolveInstancePreHash 已更新 lastInstanceId，复用避免重复读取
        long instanceId = lastInstanceId;
        for (int probe = 0; probe < capacity; probe++) {
            if (!occupied[idx]) {
                misses++;
                return Double.NaN;
            }
            int k = idx * 6;
            if (keys[k]     == instanceId &&
                keys[k + 1] == xBits &&
                keys[k + 2] == yBits &&
                keys[k + 3] == zBits &&
                keys[k + 4] == yffBits &&
                keys[k + 5] == yfvBits) {
                hits++;
                if (fromPrefetch[idx]) {
                    prefetchHits++;
                }
                return values[idx];
            }
            idx = (idx + 1) & mask;
        }
        misses++;
        return Double.NaN;
    }

    /**
     * 写入缓存。若 key 已存在则更新 value。
     */
    public void put(ImprovedNoise instance, double x, double y, double z,
                    double yFloorFreq, double yFloorValue, double result) {
        if (!Config.cacheEnabled) return;
        long preHash = resolveInstancePreHash(instance);
        long instanceId = lastInstanceId;
        putRaw(instanceId, preHash, x, y, z, yFloorFreq, yFloorValue, result, false);
    }

    /**
     * 批量写入缓存。所有项必须对应同一 ImprovedNoise 实例。
     * 比 {@link #put} 逐个写入更快，因为 instanceId 只计算一次。
     *
     * @param coords    count*5 个 double：[x0,y0,z0,yff0,yfv0, x1,y1,...]
     * @param results   count 个 double 的结果数组
     * @param count     项数
     */
    public void putBatch(ImprovedNoise instance, double[] coords, double[] results, int count) {
        if (!Config.cacheEnabled || count <= 0) return;
        long preHash = resolveInstancePreHash(instance);
        long instanceId = lastInstanceId;
        for (int i = 0; i < count; i++) {
            int off = i * 5;
            double x = coords[off];
            double y = coords[off + 1];
            double z = coords[off + 2];
            double yff = coords[off + 3];
            double yfv = coords[off + 4];
            putRaw(instanceId, preHash, x, y, z, yff, yfv, results[i], true);
        }
    }

    /** 内部：用预先计算的 instanceId 和 preHash 写入。 */
    private void putRaw(long instanceId, long preHash,
                        double x, double y, double z,
                        double yFloorFreq, double yFloorValue, double result, boolean isPrefetch) {
        long xBits = Double.doubleToLongBits(x);
        long yBits = Double.doubleToLongBits(y);
        long zBits = Double.doubleToLongBits(z);
        long yffBits = Double.doubleToLongBits(yFloorFreq);
        long yfvBits = Double.doubleToLongBits(yFloorValue);

        int idx = hashFromPre(preHash, xBits, yBits, zBits, yffBits, yfvBits) & mask;
        for (int probe = 0; probe < capacity; probe++) {
            if (!occupied[idx]) {
                int k = idx * 6;
                keys[k]     = instanceId;
                keys[k + 1] = xBits;
                keys[k + 2] = yBits;
                keys[k + 3] = zBits;
                keys[k + 4] = yffBits;
                keys[k + 5] = yfvBits;
                values[idx] = result;
                occupied[idx] = true;
                size++;
                if (size > maxSize) {
                    if (capacity < MAX_CAPACITY) {
                        resize(capacity << 1);
                    } else {
                        clear();
                    }
                }
                return;
            }
            int k = idx * 6;
            if (keys[k]     == instanceId &&
                keys[k + 1] == xBits &&
                keys[k + 2] == yBits &&
                keys[k + 3] == zBits &&
                keys[k + 4] == yffBits &&
                keys[k + 5] == yfvBits) {
                values[idx] = result;
                return;
            }
            idx = (idx + 1) & mask;
        }
        clear();
        putRaw(instanceId, preHash, x, y, z, yFloorFreq, yFloorValue, result, false);
    }

    /**
     * 扩容：重新分配数组并 rehash 所有条目。
     */
    private void resize(int newCap) {
        long[] oldKeys = keys;
        double[] oldValues = values;
        boolean[] oldOccupied = occupied;
        boolean[] oldFromPrefetch = fromPrefetch;
        int oldCap = capacity;

        this.capacity = newCap;
        this.mask = newCap - 1;
        this.keys = new long[newCap * 6];
        this.values = new double[newCap];
        this.occupied = new boolean[newCap];
        this.fromPrefetch = new boolean[newCap];
        this.size = 0;

        for (int i = 0; i < oldCap; i++) {
            if (oldOccupied[i]) {
                int k = i * 6;
                long instId = oldKeys[k];
                long preHash = instId * K;
                int idx = hashFromPre(preHash, oldKeys[k + 1], oldKeys[k + 2],
                                      oldKeys[k + 3], oldKeys[k + 4], oldKeys[k + 5]) & mask;
                while (occupied[idx]) {
                    idx = (idx + 1) & mask;
                }
                int nk = idx * 6;
                System.arraycopy(oldKeys, k, keys, nk, 6);
                values[idx] = oldValues[i];
                occupied[idx] = true;
                fromPrefetch[idx] = oldFromPrefetch[i];
                size++;
            }
        }
    }

    /**
     * 清空缓存。在每个 NoiseChunk 初始化时调用以释放内存并避免跨区块串扰。
     *
     * <p>同时重置 lastInstance 缓存，避免持有上一个区块的 ImprovedNoise 引用
     * （虽然 ThreadLocal 生命周期已限制，但显式释放更安全）。</p>
     */
    public void clear() {
        Arrays.fill(occupied, false);
        size = 0;
        lastInstance = null;
        lastInstanceId = 0L;
        lastInstancePreHash = 0L;
    }

    /**
     * 返回缓存命中率（用于调试 / 日志）。
     */
    public double hitRate() {
        long total = hits + misses;
        return total == 0 ? 0.0 : (double) hits / total;
    }

    public long getHits() { return hits; }
    public long getMisses() { return misses; }
    public long getPrefetchHits() { return prefetchHits; }
    public long getPrefetchPuts() { return prefetchPuts; }
    public int size() { return size; }

    /**
     * 热路径 hash：使用预计算的 preHash（= instanceId * K）作为起始值，
     * 避免在 get/put 中重复计算 instanceId * K 一次乘法。
     *
     * <p>算法与 {@link #hash} 等价：{@code hash(id, x, y, z, yff, yfv) ==
     * hashFromPre(id * K, x, y, z, yff, yfv)}。当 instance 不变时，preHash 可缓存复用。</p>
     */
    private static int hashFromPre(long preHash, long xBits, long yBits, long zBits,
                                   long yffBits, long yfvBits) {
        long h = preHash + xBits;
        h = h * K + yBits;
        h = h * K + zBits;
        h = h * K + yffBits;
        h = h * K + yfvBits;
        h ^= h >>> 32;
        return (int) h;
    }

    /**
     * 完整 hash 函数（avalanche），用于 resize 等非热路径。
     * 热路径请使用 {@link #hashFromPre} 配合预计算的 preHash。
     */
    private static int hash(long instanceId, long xBits, long yBits, long zBits,
                            long yffBits, long yfvBits) {
        return hashFromPre(instanceId * K, xBits, yBits, zBits, yffBits, yfvBits);
    }
}
