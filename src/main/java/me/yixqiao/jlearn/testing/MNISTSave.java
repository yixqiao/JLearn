package me.yixqiao.jlearn.testing;

import me.yixqiao.jlearn.models.Model;
import me.yixqiao.jlearn.settings.JLSettings;

/**
 * Train a model on the MNIST dataset, then save it to a file and load it back to evaluate.
 */
public class MNISTSave extends MNIST {
    /**
     * Run.
     *
     * @param args args
     */
    public static void main(String[] args) {
        JLSettings.THREAD_COUNT /= 2; // Use physical core count
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
