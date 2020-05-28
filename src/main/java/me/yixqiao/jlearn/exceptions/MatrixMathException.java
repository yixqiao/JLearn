package me.yixqiao.jlearn.exceptions;

/**
 * Exception when executing matrix math.
 */
public class MatrixMathException extends RuntimeException {
    public MatrixMathException() {
        super();
    }

    public MatrixMathException(String errorMessage) {
        super(errorMessage);
    }
}
