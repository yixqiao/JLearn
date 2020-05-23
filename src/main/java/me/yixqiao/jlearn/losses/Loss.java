package me.yixqiao.jlearn.losses;

import me.yixqiao.jlearn.core.Matrix;

import java.util.ArrayList;

public abstract class Loss {
    public abstract double getLoss(ArrayList<Matrix> output, ArrayList<Matrix> expected);

    public double getLoss(Matrix output, Matrix expected) {
        ArrayList<Matrix> outputAL = new ArrayList<>();
        outputAL.add(output);
        ArrayList<Matrix> expectedAL = new ArrayList<>();
        expectedAL.add(expected);
        return getLoss(outputAL, expectedAL);
    }
}