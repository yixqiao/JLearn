package core;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;

public class Matrix {
    int rows, cols;
    public double[][] mat;
    Random random = new Random();

    public Matrix(int rows, int cols) {
        this(rows, cols, false);
    }

    public Matrix(int rows, int cols, boolean randomize) {
        this.rows = rows;
        this.cols = cols;
        mat = new double[rows][cols];
        if (randomize) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    mat[i][j] = random.nextDouble() * 2 - 1;
                }
            }
        }
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

    public void add(Matrix m2) {
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

    public Matrix mult(int x) {
        Matrix out = clone();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out.mat[i][j] = mat[i][j] * x;
            }
        }
        return out;
    }

    public Matrix mult(Matrix m2) {
        Matrix out = clone();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out.mat[i][j] = mat[i][j] * m2.mat[i][j];
            }
        }
        return out;
    }

    public Matrix add(int x){
        Matrix out = clone();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out.mat[i][j] = mat[i][j] + x;
            }
        }
        return out;
    }

    public void sigmoid() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = 1.0 / (1.0 + Math.exp(-mat[i][j]));
            }
        }
    }

    public void relu() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = Math.max(mat[i][j], 0);
            }
        }
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
