package core;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;

public class Matrix {
    public int rows, cols;
    public double[][] mat;
    Random random = new Random();

    public Matrix(int rows, int cols) {
        this(rows, cols, 0);
    }

    public Matrix(int rows, int cols, double rFactor) {
        this.rows = rows;
        this.cols = cols;
        mat = new double[rows][cols];
        if (rFactor == 0)
            return;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = (random.nextGaussian()) * rFactor;
            }
        }
    }

    public Matrix(double[][] mat){
        this.rows = mat.length;
        this.cols = mat[0].length;
        this.mat = mat;
    }

    public Matrix dot(Matrix m2) {
        if (cols != m2.rows) {
            System.out.println("nope");
            return null;
        }
        Matrix nMatrix = new Matrix(rows, m2.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < m2.cols; j++) {
                for (int k = 0; k < cols; k++) {
                    nMatrix.mat[i][j] += mat[i][k] * m2.mat[k][j];
                }
            }
        }
        return nMatrix;
    }

    public void addIP(Matrix m2) {
        if (rows != m2.rows || cols != m2.cols) {
            System.out.println("Matrix mismatch!");
            return;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] += m2.mat[i][j];
            }
        }
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
        Matrix out = clone();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out.mat[i][j] = mat[i][j] * m2.mat[i][j];
            }
        }
        return out;
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

    public void applyEachIP(ToDoubleFunction<Double> function){
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = function.applyAsDouble(mat[i][j]);
            }
        }
    }

    public Matrix applyEach(ToDoubleFunction<Double> function){
        Matrix out = clone();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out.mat[i][j] = function.applyAsDouble(mat[i][j]);
            }
        }
        return out;
    }


    public double getMaxValue(){
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
