package me.yixqiao.jlearn.matrix;

import me.yixqiao.jlearn.exceptions.MatrixMathException;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.ToDoubleFunction;

/**
 * Matrix class and operations.
 */

public class Matrix {
    /**
     * Number of threads to use.
     */
    public static int THREAD_COUNT = Runtime.getRuntime().availableProcessors();
    /**
     * Minimum number of operations before threading is used.
     */
    public static int THREADING_MIN_OPS = (int) 1e4;

    /**
     * Contains the matrix itself.
     */
    public final double[][] mat;
    /**
     * Row count.
     */
    public int rows;

    /**
     * Column count.
     */
    public int cols;

    /**
     * Random generator.
     */
    Random random = new Random();

    /**
     * Creates a new, empty matrix.
     *
     * @param rows number of rows
     * @param cols number of columns
     */
    public Matrix(int rows, int cols) {
        this(rows, cols, 0);
    }

    /**
     * Creates a new randomized matrix.
     *
     * <p>
     *     Randomizes with <code>random.nextGaussian() * rFactor</code>
     * </p>
     *
     * @param rows number of rows
     * @param cols number of columns
     * @param rFactor factor to randomize by
     */
    public Matrix(int rows, int cols, double rFactor) {
        this.rows = rows;
        this.cols = cols;
        mat = new double[rows][cols];
        if (rFactor == 0)
            return;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] =  random.nextGaussian()* rFactor;
            }
        }
    }

    /**
     * Create a matrix from an existing 2d array of doubles.
     *
     * <p>
     *     Note that this constructor does not create a copy of the array passed in.
     * </p>
     *
     * @param mat the array to make the matrix with
     */
    public Matrix(double[][] mat) {
        this.rows = mat.length;
        this.cols = mat[0].length;
        this.mat = mat;
    }

    public Matrix dot(Matrix m2) {
        if (cols != m2.rows)
            throw new MatrixMathException(String.format("Dot mismatch of %d cols and %d rows", cols, m2.rows));

        if (THREAD_COUNT == 1 || rows * m2.cols < THREADING_MIN_OPS)
            return dot(m2, false);
        else
            return dot(m2, true);
    }

    public Matrix dot(Matrix m2, boolean useThreading) {
        if (cols != m2.rows)
            throw new MatrixMathException(String.format("Dot mismatch of %d cols and %d rows", cols, m2.rows));

        Matrix nMatrix = new Matrix(rows, m2.cols);

        if (useThreading) {
            class CalcSingle implements Runnable {
                private final int ind1, ind2;

                public CalcSingle(int ind1, int ind2) {
                    this.ind1 = ind1;
                    this.ind2 = ind2;
                }

                public void run() {
                    for (int k = 0; k < cols; k++) {
                        nMatrix.mat[ind1][ind2] += mat[ind1][k] * m2.mat[k][ind2];
                    }
                }
            }

            ExecutorService pool = Executors.newFixedThreadPool(THREAD_COUNT);

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < m2.cols; j++) {
                    Runnable r = new CalcSingle(i, j);
                    pool.execute(r);
                }
            }

            pool.shutdown();

            try {
                pool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < m2.cols; j++) {
                    for (int k = 0; k < cols; k++) {
                        nMatrix.mat[i][j] += mat[i][k] * m2.mat[k][j];
                    }
                }
            }
        }

        return nMatrix;
    }

    public void addIP(Matrix m2) {
        if (rows != m2.rows || cols != m2.cols)
            throw new MatrixMathException("Addition size mismatch");

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] += m2.mat[i][j];
            }
        }
    }

    public Matrix subtract(Matrix m2) {
        Matrix out = new Matrix(rows, cols);
        if (rows != m2.rows || cols != m2.cols)
            throw new MatrixMathException("Subtraction size mismatch");

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out.mat[i][j] = mat[i][j] - m2.mat[i][j];
            }
        }
        return out;
    }

    public Matrix multiply(double x) {
        Matrix out = clone();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out.mat[i][j] = mat[i][j] * x;
            }
        }
        return out;
    }

    public Matrix multiply(Matrix m2) {
        if (rows != m2.rows || cols != m2.cols)
            throw new MatrixMathException("Multiplication size mismatch");
        Matrix out = clone();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out.mat[i][j] = mat[i][j] * m2.mat[i][j];
            }
        }
        return out;
    }

    public void multiplyIP(double x) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] *= x;
            }
        }
    }

    public Matrix add(int x) {
        Matrix out = clone();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out.mat[i][j] = mat[i][j] + x;
            }
        }
        return out;
    }

    public void applyEachIP(ToDoubleFunction<Double> function) {
        if (THREAD_COUNT == 1 || rows * cols < THREADING_MIN_OPS)
            applyEachIP(function, false);
        else
            applyEachIP(function, true);
    }

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

            ExecutorService pool = Executors.newFixedThreadPool(THREAD_COUNT);

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Runnable r = new CalcSingle(i, j);
                    pool.execute(r);
                }
            }

            pool.shutdown();

            try {
                pool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] = function.applyAsDouble(mat[i][j]);
                }
            }
        }
    }

    public double sum() {
        double sum = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sum += mat[i][j];
            }
        }
        return sum;
    }

    public Matrix applyEach(ToDoubleFunction<Double> function) {
        if (THREAD_COUNT == 1 || rows * cols < THREADING_MIN_OPS)
            return applyEach(function, false);
        else
            return applyEach(function, true);
    }

    public Matrix applyEach(ToDoubleFunction<Double> function, boolean useThreading) {
        Matrix out = clone();
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

            ExecutorService pool = Executors.newFixedThreadPool(THREAD_COUNT);

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Runnable r = new CalcSingle(i, j);
                    pool.execute(r);
                }
            }

            pool.shutdown();

            try {
                pool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    out.mat[i][j] = function.applyAsDouble(mat[i][j]);
                }
            }
        }
        return out;
    }

    public Matrix getTranspose() {
        Matrix out = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out.mat[j][i] = mat[i][j];
            }
        }
        return out;
    }

    public double getMaxValue() {
        double max = -Double.MAX_VALUE;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                max = Math.max(max, mat[i][j]);
            }
        }
        return max;
    }

    public void randomize(double rChance, double rAmount, double rPAmount) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (random.nextDouble() < rChance) {
                    mat[i][j] *= random.nextDouble() * rAmount * 2 - rAmount + 1;
                    mat[i][j] += random.nextDouble() * rPAmount * 2 - rPAmount;
                }
            }
        }
    }

    public void writeToFile(DataOutputStream dos) {
        try {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    dos.writeDouble(mat[i][j]);
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void readFromFile(DataInputStream dis) {
        try {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] = dis.readDouble();
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void crossOver(Matrix m2, double weightSelf) {
        if (rows != m2.rows || cols != m2.cols)
            throw new MatrixMathException("Cross over size mismatch");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (random.nextDouble() > weightSelf) {
                    mat[i][j] = m2.mat[i][j];
                }
            }
        }
    }

    public void printMatrix() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.print(mat[i][j]);
                System.out.print(" ");
            }
            System.out.println();
        }
    }

    public Matrix clone() {
        Matrix out = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out.mat[i][j] = mat[i][j];
            }
        }
        return out;
    }
}
