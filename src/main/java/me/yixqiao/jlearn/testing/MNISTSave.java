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
        mnist.evaluateModel();
    }

    protected void loadModel() {
        this.model = Model.readFromFile("m.jlm");
        System.out.println("Finished loading model.");
    }

    @Override
    protected void afterTrain() {
        model.saveToFile("m.jlm");
        System.out.println("Finished saving model.\n");
    }
}
