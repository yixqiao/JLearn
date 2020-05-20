package losses;

import core.Matrix;
import models.Model;

import java.util.ArrayList;

public abstract class Loss {
    public abstract double getLoss(Model model, ArrayList<Matrix> input, ArrayList<Matrix> expected);

    public double getLoss(Model model, Matrix input, Matrix expected) {
        ArrayList<Matrix> inputAL = new ArrayList<>();
        inputAL.add(input);
        ArrayList<Matrix> expectedAL = new ArrayList<>();
        expectedAL.add(expected);
        return getLoss(model, inputAL, expectedAL);
    }
}
