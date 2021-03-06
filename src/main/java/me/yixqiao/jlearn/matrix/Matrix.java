package me.yixqiao.jlearn.matrix;

import me.yixqiao.jlearn.exceptions.MatrixMathException;
import me.yixqiao.jlearn.settings.JLSettings;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.ToDoubleFunction;

/**
 * Matrix class and operations.
 */

public class Matrix implements Serializable {
    private static final Random random = new Random();
    /**
     * Contains the matrix itself.
     */
    public final double[][] mat;
    /**
     * Row count.
     */
    public final int rows;
    /**
     * Column count.
     */
    public final int cols;

    /**
     * Create a new matrix using an <code>Matrix.Init</code> object.
     *
     * @param rows number of rows
     * @param cols number of columns
     * @param init instance of init object
     */
    public Matrix(int rows, int cols, Init init) {
        this(rows, cols);
        init.apply(this);
    }

    /**
     * Creates a new, empty matrix.
     *
     * @param rows number of rows
     * @param cols number of columns
     */
    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        mat = new double[rows][cols];
    }

    /**
     * Create a matrix from an existing 2d array of doubles.
     *
     * <p>
     * Note that this constructor does not create a copy of the array passed in.
     * </p>
     *
     * @param mat the array to make the matrix with
     */
    public Matrix(double[][] mat) {
        this.rows = mat.length;
        this.cols = mat[0].length;
        this.mat = mat;
    }

    /**
     * Calculate the dot product.
     *
     * @param m2 the other matrix
     * @return the result
     */
    public Matrix dot(Matrix m2) {
        if (cols != m2.rows)
            throw new MatrixMathException(String.format("Dot mismatch of %d cols and %d rows", cols, m2.rows));

        if (JLSettings.THREAD_COUNT == 1 || cols * m2.rows < JLSettings.THREADING_MIN_OPS)
            return dot(m2, false);
        else
            return dot(m2, true);
    }

    /**
     * Calculate the dot product.
     *
     * @param m2           the other matrix
     * @param useThreading whether to use threading
     * @return the result
     */
    public Matrix dot(Matrix m2, boolean useThreading) {
        if (cols != m2.rows)
            throw new MatrixMathException(String.format("Dot mismatch of %d cols and %d rows", cols, m2.rows));

        Matrix nMatrix = new Matrix(rows, m2.cols);

        if (useThreading) {
            class CalcSingle implements Runnable {
                private final int r;

                public CalcSingle(int r) {
                    this.r = r;
                }

                public void run() {
                    for (int i = 0; i < cols; i++) {
                        for (int c = 0; c < m2.cols; c++) {
                            nMatrix.mat[r][c] += mat[r][i] * m2.mat[i][c];
                        }
                    }
                }
            }

            ExecutorService pool = Executors.newFixedThreadPool(JLSettings.THREAD_COUNT);

            for (int r = 0; r < rows; r++) {
                Runnable rn = new CalcSingle(r);
                pool.execute(rn);
            }

            pool.shutdown();

            try {
                pool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            for (int r = 0; r < rows; r++) {
                for (int i = 0; i < cols; i++) {
                    for (int c = 0; c < m2.cols; c++) { // Swapped (https://stackoverflow.com/a/4300744)
                        nMatrix.mat[r][c] += mat[r][i] * m2.mat[i][c];
                    }
                }
            }
        }

        return nMatrix;
    }

    /**
     * Add a scalar to all elements.
     *
     * @param x number to add
     * @return new matrix
     */
    public Matrix add(double x) {
        Matrix out = cloneMatrix();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                out.mat[r][c] = mat[r][c] + x;
            }
        }
        return out;
    }

    /**
     * Add a scalar to all elements, in place.
     *
     * @param x number to add
     */
    public void addIP(double x) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                mat[r][c] += x;
            }
        }
    }

    /**
     * Add another matrix element by element.
     *
     * @param m2 other matrix with same dimensions
     * @return new matrix
     */
    public Matrix add(Matrix m2) {
        if (rows != m2.rows || cols != m2.cols)
            throw new MatrixMathException(String.format("Addition size mismatch of %dx%d and %dx%d", rows, cols, m2.rows, m2.cols));

        Matrix out = cloneMatrix();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                out.mat[r][c] += m2.mat[r][c];
            }
        }
        return out;
    }

    /**
     * Add another matrix element by element, in place.
     *
     * @param m2 other matrix with same dimensions
     */
    public void addIP(Matrix m2) {
        if (rows != m2.rows || cols != m2.cols)
            throw new MatrixMathException(String.format("Addition size mismatch of %dx%d and %dx%d", rows, cols, m2.rows, m2.cols));

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                mat[r][c] += m2.mat[r][c];
            }
        }
    }

    /**
     * Subtract a scalar from all elements.
     *
     * @param x number to subtract
     * @return new matrix
     */
    public Matrix subtract(double x) {
        Matrix out = cloneMatrix();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                out.mat[r][c] -= x;
            }
        }
        return out;
    }

    /**
     * Subtract another matrix element by element.
     *
     * @param m2 other matrix with the same dimensions
     * @return the resulting matrix
     */
    public Matrix subtract(Matrix m2) {
        Matrix out = new Matrix(rows, cols);
        if (rows != m2.rows || cols != m2.cols)
            throw new MatrixMathException(String.format("Subtraction size mismatch of %dx%d and %dx%d", rows, cols, m2.rows, m2.cols));

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                out.mat[r][c] = mat[r][c] - m2.mat[r][c];
            }
        }
        return out;
    }

    /**
     * Multiply each matrix element by a scalar.
     *
     * @param x the number to multiply by
     * @return the resulting matrix
     */
    public Matrix multiply(double x) {
        Matrix out = cloneMatrix();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                out.mat[r][c] = mat[r][c] * x;
            }
        }
        return out;
    }

    /**
     * Multiply each matrix element by a scalar, in place.
     *
     * @param x the number to multiply by
     */
    public void multiplyIP(double x) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                mat[r][c] *= x;
            }
        }
    }

    /**
     * Multiply by another matrix element by element.
     *
     * @param m2 other matrix with the same dimensions
     * @return the resulting matrix
     */
    public Matrix multiply(Matrix m2) {
        if (rows != m2.rows || cols != m2.cols)
            throw new MatrixMathException(String.format("Multiplication size mismatch of %dx%d and %dx%d", rows, cols, m2.rows, m2.cols));
        Matrix out = cloneMatrix();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                out.mat[r][c] = mat[r][c] * m2.mat[r][c];
            }
        }
        return out;
    }

    /**
     * Multiply by another matrix element by element, in place.
     *
     * @param m2 other matrix with the same dimensions
     */
    public void multiplyIP(Matrix m2) {
        if (rows != m2.rows || cols != m2.cols)
            throw new MatrixMathException(String.format("Multiplication size mismatch of %dx%d and %dx%d", rows, cols, m2.rows, m2.cols));
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                mat[r][c] *= m2.mat[r][c];
            }
        }
    }

    /**
     * Divide by another matrix element by element, in place.
     *
     * @param m2 other matrix
     */
    public void divideIP(Matrix m2) {
        if (rows != m2.rows || cols != m2.cols)
            throw new MatrixMathException(String.format("Division size mismatch of %dx%d and %dx%d", rows, cols, m2.rows, m2.cols));
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                mat[r][c] /= m2.mat[r][c];
            }
        }
    }

    /**
     * Apply a function to each element.
     *
     * @param function the function to apply
     * @return the resulting matrix
     */
    public Matrix applyEach(ToDoubleFunction<Double> function) {
        // if (JLSettings.THREAD_COUNT == 1 || rows * cols < JLSettings.THREADING_MIN_OPS)
        //     return applyEach(function, false);
        // else
        //     return applyEach(function, true);
        return applyEach(function, false);
    }

    /**
     * Apply a function to each element.
     *
     * @param function     the function to apply
     * @param useThreading whether to use threading
     * @return the resulting matrix
     */
    public Matrix applyEach(ToDoubleFunction<Double> function, boolean useThreading) {
        Matrix out = cloneMatrix();
        if (useThreading) {
            class CalcSingle implements Runnable {
                private final int r, c;

                public CalcSingle(int r, int c) {
                    this.r = r;
                    this.c = c;
                }

                public void run() {
                    out.mat[r][c] = function.applyAsDouble(mat[r][c]);
                }
            }

            ExecutorService pool = Executors.newFixedThreadPool(JLSettings.THREAD_COUNT);

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    Runnable rn = new CalcSingle(r, c);
                    pool.execute(rn);
                }
            }

            pool.shutdown();

            try {
                pool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    out.mat[r][c] = function.applyAsDouble(mat[r][c]);
                }
            }
        }
        return out;
    }

    /**
     * Apply a function to each element, in place.
     *
     * @param function the function to apply
     */
    public void applyEachIP(ToDoubleFunction<Double> function) {
        // if (JLSettings.THREAD_COUNT == 1 || rows * cols < JLSettings.THREADING_MIN_OPS)
        //     applyEachIP(function, false);
        // else
        //     applyEachIP(function, true);
        applyEachIP(function, false);
    }

    /**
     * Apply a function to each element, in place.
     *
     * @param function     the function to apply
     * @param useThreading whether to use threading
     */
    public void applyEachIP(ToDoubleFunction<Double> function, boolean useThreading) {
        if (useThreading) {
            class CalcSingle implements Runnable {
                private final int r, c;

                public CalcSingle(int r, int c) {
                    this.r = r;
                    this.c = c;
                }

                public void run() {
                    mat[r][c] = function.applyAsDouble(mat[r][c]);
                }
            }

            ExecutorService pool = Executors.newFixedThreadPool(JLSettings.THREAD_COUNT);

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    Runnable rn = new CalcSingle(r, c);
                    pool.execute(rn);
                }
            }

            pool.shutdown();

            try {
                pool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    mat[r][c] = function.applyAsDouble(mat[r][c]);
                }
            }
        }
    }

    /**
     * Get the sum of the elements.
     *
     * @return the sum
     */
    public double getSum() {
        double sum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                sum += mat[r][c];
            }
        }
        return sum;
    }

    /**
     * Get the greatest value in the matrix.
     *
     * @return the max value
     */
    public double getMaxValue() {
        double max = -Double.MAX_VALUE;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                max = Math.max(max, mat[r][c]);
            }
        }
        return max;
    }

    /**
     * Get the matrix transpose.
     *
     * @return the transposed matrix
     */
    public Matrix getTranspose() {
        Matrix out = new Matrix(cols, rows);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                out.mat[c][r] = mat[r][c];
            }
        }
        return out;
    }

    /**
     * Write the matrix to a file.
     *
     * @param dos stream to write to
     */
    public void writeToFile(DataOutputStream dos) {
        try {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    dos.writeDouble(mat[r][c]);
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Read the matrix from a file.
     *
     * @param dis stream to read from
     */
    public void readFromFile(DataInputStream dis) {
        try {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    mat[r][c] = dis.readDouble();
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Cross over with another matrix.
     *
     * @param m2         other matrix
     * @param weightSelf weight given to itself
     * @deprecated As of v0.2.0, this method has no use.
     */
    @Deprecated
    public void crossOver(Matrix m2, double weightSelf) {
        if (rows != m2.rows || cols != m2.cols)
            throw new MatrixMathException(String.format("Cross over size mismatch of %dx%d and %dx%d", rows, cols, m2.rows, m2.cols));
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (random.nextDouble() > weightSelf) {
                    mat[r][c] = m2.mat[r][c];
                }
            }
        }
    }

    /**
     * Print out the matrix to stdout.
     */
    public void printMatrix() {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                System.out.print(mat[r][c]);
                System.out.print(" ");
            }
            System.out.println();
        }
    }

    /**
     * Create a clone of the matrix.
     *
     * @return the clone
     */
    public Matrix cloneMatrix() {
        Matrix out = new Matrix(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                out.mat[r][c] = mat[r][c];
            }
        }
        return out;
    }

    /**
     * Initialization methods for Matrix.
     */
    public abstract static class Init {
        /**
         * Apply the initialization (in place) to a new matrix.
         *
         * @param m the matrix
         */
        public abstract void apply(Matrix m);

        /**
         * Init all with 0s.
         */
        public static class Empty extends Init {
            @Override
            public void apply(Matrix m) {
                // Do nothing, as matrix should already be initialized with 0
            }
        }

        /**
         * Init all with a number.
         */
        public static class Fill extends Init {
            double fNum;

            /**
             * Create a new class.
             *
             * @param fNum the number to fill the matrix with
             */
            public Fill(double fNum) {
                this.fNum = fNum;
            }

            @Override
            public void apply(Matrix m) {
                m.applyEachIP(x -> fNum);
            }
        }

        /**
         * Generate a uniform range of random numbers to fill the matrix.
         * <p>
         * The uniform will be centered around <code>center</code>,
         * extending for <code>range</code> in either direction.
         * </p>
         */
        public static class Uniform extends Init {
            double center;
            double range;

            /**
             * Create a new class.
             *
             * @param center center of the uniform
             * @param range  range in either direction
             */
            public Uniform(double center, double range) {
                this.center = center;
                this.range = range;
            }

            /**
             * Create a new class.
             *
             * @param range range in either direction
             */
            public Uniform(double range) {
                this(0, range);
            }

            @Override
            public void apply(Matrix m) {
                m.applyEachIP(x -> (random.nextDouble() * 2 - 1) * range + center);
            }
        }

        /**
         * Generate random numbers from the gaussian distribution to fill the matrix.
         * <p>
         * The numbers will be centered around <code>center</code>,
         * and have a standard deviation of <code>range</code> direction.
         * </p>
         */
        public static class Gaussian extends Init {
            double center;
            double deviation;

            /**
             * Create a new class.
             *
             * @param center    center of the gaussian
             * @param deviation standard deviation
             */
            public Gaussian(double center, double deviation) {
                this.center = center;
                this.deviation = deviation;
            }

            /**
             * Create a new class.
             *
             * @param deviation standard deviation
             */
            public Gaussian(double deviation) {
                this(0, deviation);
            }

            @Override
            public void apply(Matrix m) {
                m.applyEachIP(x -> random.nextGaussian() * deviation + center);
            }
        }
    }
}
