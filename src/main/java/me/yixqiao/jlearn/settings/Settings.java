package me.yixqiao.jlearn.settings;

/**
 * Global settings for JLearn.
 */
public final class Settings {
    /**
     * Number of threads to use.
     */
    public static int THREAD_COUNT = Runtime.getRuntime().availableProcessors();
    /**
     * Minimum number of operations before threading is used.
     */
    public static int THREADING_MIN_OPS = (int) 1000;
}
