package losses;

import core.Matrix;
import models.Model;

import java.util.ArrayList;

public abstract class OutputLoss extends Loss {
    public abstract double getOutputLoss(ArrayList<Matrix> output, ArrayList<Matrix> expected);

    @Override
    public double getLoss(Model model, ArrayList<Matrix> input, ArrayList<Matrix> expected) {
        ArrayList<Matrix> output = new ArrayList<>();
        for (Matrix in : input)
            output.add(model.forwardPropagate(in));
        return getOutputLoss(output, expected);
    }
}
