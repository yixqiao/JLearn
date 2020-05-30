package me.yixqiao.jlearn.testing;

import me.yixqiao.jlearn.metrics.Accuracy;
import me.yixqiao.jlearn.metrics.Metric;
import me.yixqiao.jlearn.models.Model;

import java.util.ArrayList;

public class MNISTSave extends MNIST {
    public static void main(String[] args) {
        runTrain();
        runLoad();
    }

    protected static void runTrain() {
        MNISTSave mnist = new MNISTSave();
        // mnist.writeDataset();
        mnist.initInputs();
        mnist.buildModel();
        mnist.train();
    }

    protected static void runLoad() {
        MNISTSave mnist = new MNISTSave();
        mnist.loadModel();
        mnist.initInputs();
        mnist.printPredictions();
    }

    protected void loadModel() {
        this.model = Model.readFromFile("m.tmp");
        System.out.println("Finished loading model.");
    }

    @Override
    protected void train() {
        printPredictions();

        ArrayList<Metric> metrics = new ArrayList<>() {{
            add(new Accuracy());
        }};
        model.fit(inputs, outputs, evalInputs, evalOutputs, 0.01, 4, 2, 1, metrics);

        printPredictions();

        model.saveToFile("m.tmp");

        System.out.println("Finished saving model.\n");
    }
}
