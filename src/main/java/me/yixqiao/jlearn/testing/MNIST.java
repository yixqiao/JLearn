package me.yixqiao.jlearn.testing;

import me.yixqiao.jlearn.activations.*;
import me.yixqiao.jlearn.datasets.DatasetTT;
import me.yixqiao.jlearn.datasets.MNISTDigits;
import me.yixqiao.jlearn.matrix.Matrix;
import me.yixqiao.jlearn.layers.Dense;
import me.yixqiao.jlearn.layers.InputLayer;
import me.yixqiao.jlearn.losses.CrossEntropy;
import me.yixqiao.jlearn.metrics.Accuracy;
import me.yixqiao.jlearn.metrics.Metric;
import me.yixqiao.jlearn.models.Model;
import me.yixqiao.jlearn.settings.JLSettings;

import java.util.ArrayList;

/**
 * Train a network on the MNIST digits dataset.
 */
public class MNIST {
    /**
     * Inputs from dataset.
     */
    protected Matrix inputs;
    /**
     * Outputs from dataset.
     */
    protected Matrix outputs;
    /**
     * Test inputs from dataset.
     */
    protected Matrix evalInputs;
    /**
     * Test outputs from dataset.
     */
    protected Matrix evalOutputs;
    /**
     * Model.
     */
    protected Model model;

    /**
     * Run.
     *
     * @param args args
     */
    public static void main(String[] args) {
        JLSettings.THREAD_COUNT /= 2; // Use physical core count

        MNIST mnist = new MNIST();
        mnist.initInputs();
        mnist.buildModel();
        mnist.train();
    }

    /**
     * Initialize inputs.
     */
    protected void initInputs() {
        DatasetTT data = MNISTDigits.load();
        inputs = data.train.x;
        outputs = data.train.y;
        evalInputs = data.test.x;
        evalOutputs = data.test.y;
    }

    /**
     * Build model.
     */
    protected void buildModel() {
        model = new Model();
        model.addLayer(new InputLayer(28 * 28))
                .addLayer(new Dense(64, new ReLU()))
                .addLayer(new Dense(32, new ReLU()))
                .addLayer(new Dense(10, new Softmax()));

        model.buildModel(new CrossEntropy());

        model.printSummary();
    }

    /**
     * Train model.
     */
    protected void train() {
        // printPredictions();

        ArrayList<Metric> metrics = new ArrayList<>() {{
            add(new Accuracy());
        }};

        model.fit(new Model.FitBuilder(inputs, outputs)
                .learningRate(0.005)
                .batchSize(64)
                .epochs(10)
                .metrics(metrics)
                .eval(evalInputs, evalOutputs)
        );

        // printPredictions();

        System.out.println();
        evaluateModel();

        afterTrain();
    }

    /**
     * Evaluate model.
     */
    protected void evaluateModel() {
        ArrayList<Metric> metrics = new ArrayList<Metric>() {{
            add(new Accuracy());
        }};
        System.out.print("Eval: ");
        model.evaluate(evalInputs, evalOutputs, metrics);
    }

    /**
     * Method to be overriden.
     */
    protected void afterTrain() {
        // Override in subclass
    }

    /**
     * Print predictions.
     */
    protected void printPredictions() {
        model.comparePredictions(inputs, outputs, 5);
    }
}

