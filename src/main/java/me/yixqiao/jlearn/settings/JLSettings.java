package me.yixqiao.jlearn.settings;

/**
 * Global settings for JLearn.
 */
public final class JLSettings {
    /**
     * Number of threads to use.
     */
    public static int THREAD_COUNT = Runtime.getRuntime().availableProcessors();
    /**
     * Minimum number of operations before threading is used.
     * <p>
     * Note: generally implemented as number of operations within each thread.
     * </p>
     */
    public static int THREADING_MIN_OPS = (int) 1000;
}
