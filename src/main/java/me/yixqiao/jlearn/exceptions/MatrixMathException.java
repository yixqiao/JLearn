package me.yixqiao.jlearn.exceptions;

public class MatrixMathException extends RuntimeException {
    public MatrixMathException() {
        super();
    }

    public MatrixMathException(String errorMessage) {
        super(errorMessage);
    }
}
