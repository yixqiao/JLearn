package me.yixqiao.jlearn.testing;

import me.yixqiao.jlearn.metrics.Accuracy;
import me.yixqiao.jlearn.metrics.Metric;
import me.yixqiao.jlearn.models.Model;
import me.yixqiao.jlearn.settings.Settings;

import java.util.ArrayList;

public class MNISTSave extends MNIST {
    /**
     * Run.
     *
     * @param args args
     */
    public static void main(String[] args) {
        Settings.THREAD_COUNT /= 2; // Use physical core count
        runTrain();
        runLoad();
    }

    /**
     * Train a new model.
     */
    protected static void runTrain() {
        MNISTSave mnist = new MNISTSave();
        // mnist.writeDataset();
        mnist.initInputs();
        mnist.buildModel();
        mnist.train();
    }

    /**
     * Load and evaluate a model.
     */
    protected static void runLoad() {
        MNISTSave mnist = new MNISTSave();
        mnist.loadModel();
        mnist.initInputs();
        mnist.evaluateModel();
    }

    /**
     * Load the model.
     */
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
