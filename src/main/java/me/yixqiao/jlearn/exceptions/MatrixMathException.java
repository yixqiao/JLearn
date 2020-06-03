package me.yixqiao.jlearn.exceptions;

/**
 * Exception when executing matrix math.
 */
public class MatrixMathException extends RuntimeException {
    /**
     * Create an exception.
     */
    public MatrixMathException() {
        super();
    }

    /**
     * Create an exception.
     *
     * @param errorMessage custom error message
     */
    public MatrixMathException(String errorMessage) {
        super(errorMessage);
    }
}
