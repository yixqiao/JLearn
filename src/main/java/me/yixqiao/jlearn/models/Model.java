package me.yixqiao.jlearn.models;

import me.yixqiao.jlearn.exceptions.NeuralNetworkException;
import me.yixqiao.jlearn.layers.InputLayer;
import me.yixqiao.jlearn.matrix.Matrix;
import me.yixqiao.jlearn.layers.Layer;
import me.yixqiao.jlearn.losses.Loss;
import me.yixqiao.jlearn.metrics.Metric;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Neural network model.
 */

public class Model implements Serializable {
    protected final ArrayList<Layer> layers;
    protected int layerCount;
    protected Loss loss;

    /**
     * Create a new model.
     */
    public Model() {
        layers = new ArrayList<>();
    }

    /**
     * Read a model from a file.
     *
     * @param filePath path to the file
     * @return the model read from the file
     */
    public static Model readFromFile(String filePath) {
        Model m = null;
        try {
            FileInputStream fis = new FileInputStream(filePath);
            GZIPInputStream gzipIn = new GZIPInputStream(fis);
            ObjectInputStream ois = new ObjectInputStream(gzipIn);

            m = (Model) ois.readObject();

            fis.close();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return m;
    }

    /**
     * Add a layer to the model.
     *
     * @param layer the layer instance to add
     * @return the model itself to allow for daisy chaining
     */
    public Model addLayer(Layer layer) {
        layers.add(layer);
        return this;
    }

    /**
     * Build the model. Run this after adding layers and before training.
     *
     * @param loss an instance of the loss function to use in the model
     */
    public void buildModel(Loss loss) {
        layerCount = layers.size();
        if (!(layers.get(0) instanceof InputLayer)) {
            throw new NeuralNetworkException("First layer is not an input layer");
        }
        for (int layer = 0; layer < layerCount - 1; layer++) {
            layers.get(layer + 1).initLayer(layers.get(layer).getOutSize(), layers.get(layer).getActivation());
        }
        this.loss = loss;
    }

    public void printSummary() {
        System.out.printf("\nModel with %d layers:\n", layerCount);
        for (Layer l : layers) {
            System.out.println(l);
        }
        System.out.println();
    }

    /**
     * Train the model on data.
     *
     * @param input        input data to train on
     * @param expected     expected outputs
     * @param evalInput    input of evaluation set
     * @param evalExpected expected outputs of evaluation set
     * @param learningRate learning rate of training
     * @param batchSize    size of each minibatch
     * @param epochs       number of epochs to train for
     * @param logInterval  log every n epochs
     * @param metrics      metrics to display
     */
    public void fit(Matrix input, Matrix expected, Matrix evalInput, Matrix evalExpected,
                    double learningRate, int batchSize, int epochs, int logInterval, ArrayList<Metric> metrics) {
        int totalSamples = input.rows;
        ArrayList<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalSamples; i++) indices.add(i);
        learningRate *= Math.sqrt(batchSize);

        ArrayList<Matrix> errors;
        if (metrics == null)
            metrics = new ArrayList<>();


        for (int epoch = 0; epoch < epochs; epoch++) {
            double lossA = 0;
            double[] metricA = new double[metrics.size()];

            long epochStart = System.nanoTime();

            Collections.shuffle(indices);

            FitPrint fp = null;
            int maxLineLen = 0;
            if ((epoch + 1) % logInterval == 0) {
                fp = new FitPrint();
                fp.start();
            }

            for (int batchNum = 0; batchNum < totalSamples / batchSize; batchNum++) {
                Matrix batchInput = new Matrix(batchSize, input.cols);
                Matrix batchExpected = new Matrix(batchSize, expected.cols);
                for (int i = 0; i < batchSize; i++) {
                    for (int j = 0; j < input.cols; j++) {
                        batchInput.mat[i][j] = input.mat[indices.get(batchNum * batchSize + i)][j];
                    }
                    for (int j = 0; j < expected.cols; j++) {
                        batchExpected.mat[i][j] = expected.mat[indices.get(batchNum * batchSize + i)][j];
                    }
                }
                //System.out.println("\n\n\n\n\n\n\n\n\n\n\n");
                //batchInput.printMatrix();
                Matrix batchOutput = forwardPropagate(batchInput);
                // System.out.println(batchNum + " " + batchOutput.mat[0][0]);
                if ((epoch + 1) % logInterval == 0) {
                    lossA += getLoss(loss, batchOutput, batchExpected);
                    for (int i = 0; i < metrics.size(); i++)
                        metricA[i] += getMetric(metrics.get(i), batchOutput, batchExpected);
                }

                errors = backPropagate(batchExpected);
                update(errors, learningRate);

                if ((epoch + 1) % logInterval == 0) {
                    String fitOutput = "";

                    double timeElapsed = (double) (System.nanoTime() - epochStart) / 1e9;

                    fitOutput += String.format("\rE: %d - T: %5.2fs - L: %.5f", epoch + 1, timeElapsed, lossA / (batchNum + 1));

                    for (int i = 0; i < metrics.size(); i++)
                        fitOutput += String.format((" - " + metrics.get(i).getFormatString()), metricA[i] / (batchNum + 1));

                    fitOutput += " - [";

                    int progress = (int) ((double) batchNum / ((double) totalSamples / batchSize) * 20);


                    for (int i = 0; i < progress; i++) {
                        fitOutput += "=";
                    }

                    fitOutput += ">";

                    for (int i = 0; i < 20 - progress - 1; i++) {
                        fitOutput += ".";
                    }

                    fitOutput += "]";

                    maxLineLen = Math.max(maxLineLen, fitOutput.length());

                    fp.setOutput(fitOutput);
                }
            }

            if ((epoch + 1) % logInterval == 0) {
                fp.stopThread();

                try {
                    fp.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                String toPrint = String.format("\rE: %d - T: %5.2fs - L: %.5f", epoch + 1, (double) (System.nanoTime() - epochStart) / 1e9,
                        lossA / (double) (totalSamples / batchSize));

                for (int i = 0; i < metrics.size(); i++)
                    toPrint += String.format((" - " + metrics.get(i).getFormatString()), metricA[i] / (double) (totalSamples / batchSize));

                toPrint += " - Evaluating...";

                for (int i = toPrint.length(); i < maxLineLen; i++)
                    toPrint += " ";

                System.out.print(toPrint);

                // Matrix output = forwardPropagate(input);
                // output.printMatrix();
                lossA /= (double) (totalSamples / batchSize);
                for (int i = 0; i < metrics.size(); i++)
                    metricA[i] /= (double) (totalSamples / batchSize);
                Matrix evalOutput = forwardPropagate(evalInput);

                double timeElapsed = (double) (System.nanoTime() - epochStart) / 1e9;

                String finalOutput = String.format("\rE: %d - T: %5.2fs - L: %.5f", epoch + 1, timeElapsed, lossA);

                for (int i = 0; i < metrics.size(); i++)
                    finalOutput += String.format((" - " + metrics.get(i).getFormatString()), metricA[i]);

                double evalLoss = getLoss(loss, evalOutput, evalExpected);
                finalOutput += String.format(" - EL: %.5f", evalLoss);

                for (Metric metric : metrics)
                    finalOutput += String.format((" - E" + metric.getFormatString()),
                            getMetric(metric, evalOutput, evalExpected));

                for (int i = 0; i < 3; i++) {
                    System.out.print(finalOutput);
                    System.out.flush();
                }

                System.out.println();
            }
        }
    }

    public void fit(FitBuilder fb) {
        Matrix input = fb.input;
        Matrix expected = fb.expected;
        double learningRate = fb.learningRate;
        ArrayList<Metric> metrics = fb.metrics;
        int batchSize = fb.batchSize;
        int epochs = fb.epochs;
        int logInterval = fb.logInterval;


        int totalSamples = input.rows;
        ArrayList<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalSamples; i++) indices.add(i);
        learningRate *= Math.sqrt(batchSize);

        ArrayList<Matrix> errors;


        for (int epoch = 0; epoch < epochs; epoch++) {
            double lossA = 0;
            double[] metricA = new double[metrics.size()];

            long epochStart = System.nanoTime();

            Collections.shuffle(indices);

            FitPrint fp = null;
            int maxLineLen = 0;
            if ((epoch + 1) % logInterval == 0) {
                fp = new FitPrint();
                fp.start();
            }

            for (int batchNum = 0; batchNum < totalSamples / batchSize; batchNum++) {
                Matrix batchInput = new Matrix(batchSize, input.cols);
                Matrix batchExpected = new Matrix(batchSize, expected.cols);
                for (int i = 0; i < batchSize; i++) {
                    for (int j = 0; j < input.cols; j++) {
                        batchInput.mat[i][j] = input.mat[indices.get(batchNum * batchSize + i)][j];
                    }
                    for (int j = 0; j < expected.cols; j++) {
                        batchExpected.mat[i][j] = expected.mat[indices.get(batchNum * batchSize + i)][j];
                    }
                }
                //System.out.println("\n\n\n\n\n\n\n\n\n\n\n");
                //batchInput.printMatrix();
                Matrix batchOutput = forwardPropagate(batchInput);
                // System.out.println(batchNum + " " + batchOutput.mat[0][0]);
                if ((epoch + 1) % logInterval == 0) {
                    lossA += getLoss(loss, batchOutput, batchExpected);
                    for (int i = 0; i < metrics.size(); i++)
                        metricA[i] += getMetric(metrics.get(i), batchOutput, batchExpected);
                }

                errors = backPropagate(batchExpected);
                update(errors, learningRate);

                if ((epoch + 1) % logInterval == 0) {
                    String fitOutput = "";

                    double timeElapsed = (double) (System.nanoTime() - epochStart) / 1e9;

                    fitOutput += String.format("\rE: %d - T: %5.2fs - L: %.5f", epoch + 1, timeElapsed, lossA / (batchNum + 1));

                    for (int i = 0; i < metrics.size(); i++)
                        fitOutput += String.format((" - " + metrics.get(i).getFormatString()), metricA[i] / (batchNum + 1));

                    fitOutput += " - [";

                    int progress = (int) ((double) batchNum / ((double) totalSamples / batchSize) * 20);


                    for (int i = 0; i < progress; i++) {
                        fitOutput += "=";
                    }

                    fitOutput += ">";

                    for (int i = 0; i < 20 - progress - 1; i++) {
                        fitOutput += ".";
                    }

                    fitOutput += "]";

                    maxLineLen = Math.max(maxLineLen, fitOutput.length());

                    fp.setOutput(fitOutput);
                }
            }

            if ((epoch + 1) % logInterval == 0) {
                fp.stopThread();

                try {
                    fp.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                String toPrint = String.format("\rE: %d - T: %5.2fs - L: %.5f", epoch + 1, (double) (System.nanoTime() - epochStart) / 1e9,
                        lossA / (double) (totalSamples / batchSize));

                for (int i = 0; i < metrics.size(); i++)
                    toPrint += String.format((" - " + metrics.get(i).getFormatString()), metricA[i] / (double) (totalSamples / batchSize));

                toPrint += " - Evaluating...";

                for (int i = toPrint.length(); i < maxLineLen; i++)
                    toPrint += " ";

                System.out.print(toPrint);

                // Matrix output = forwardPropagate(input);
                // output.printMatrix();
                lossA /= (double) (totalSamples / batchSize);
                for (int i = 0; i < metrics.size(); i++)
                    metricA[i] /= (double) (totalSamples / batchSize);

                Matrix evalOutput = null;
                if (fb.hasEval)
                    evalOutput = forwardPropagate(fb.evalInput);

                double timeElapsed = (double) (System.nanoTime() - epochStart) / 1e9;

                String finalOutput = String.format("\rE: %d - T: %5.2fs - L: %.5f", epoch + 1, timeElapsed, lossA);

                for (int i = 0; i < metrics.size(); i++)
                    finalOutput += String.format((" - " + metrics.get(i).getFormatString()), metricA[i]);

                if (fb.hasEval) {
                    double evalLoss = getLoss(loss, evalOutput, fb.evalExpected);
                    finalOutput += String.format(" - EL: %.5f", evalLoss);

                    for (Metric metric : metrics)
                        finalOutput += String.format((" - E" + metric.getFormatString()),
                                getMetric(metric, evalOutput, fb.evalExpected));
                }

                for (int i = 0; i < 3; i++) {
                    System.out.print(finalOutput);
                    System.out.flush();
                }

                System.out.println();
            }
        }
    }

    /**
     * Train on a single batch of input and output.
     *
     * @param input        input data to train on
     * @param expected     expected outputs
     * @param learningRate learning rate of training
     */
    public void trainOnBatch(Matrix input, Matrix expected, double learningRate) {
        forwardPropagate(input);
        ArrayList<Matrix> errors = backPropagate(expected);
        update(errors, learningRate);
    }

    /**
     * Predict on a batch of input.
     *
     * @param input input data to feed forward
     * @return prediction of model for input
     */
    public Matrix predict(Matrix input) {
        return forwardPropagate(input);
    }

    /**
     * Evaluate the performance of the model.
     *
     * @param input    input data
     * @param expected expected outputs
     * @param metrics  metrics to display
     */
    public void evaluate(Matrix input, Matrix expected, ArrayList<Metric> metrics) {
        Matrix output = forwardPropagate(input);

        double evalLoss = getLoss(loss, output, expected);
        System.out.printf("L: %.5f", evalLoss);

        for (int i = 0; i < metrics.size(); i++)
            System.out.printf((", " + metrics.get(i).getFormatString()),
                    getMetric(metrics.get(i), output, expected));

        System.out.println();
    }

    /**
     * Forward propagate a batch of input.
     *
     * @param input input data to feed forward
     * @return result of model for input
     */
    public Matrix forwardPropagate(Matrix input) {
        Matrix activations = input.clone();
        for (int layerNum = 0; layerNum < layerCount; layerNum++) {
            activations = layers.get(layerNum).forwardPropagate(activations);
        }
        return activations;
    }

    /**
     * Backpropagate model after forward propagating input.
     *
     * @param expected expected output for model
     * @return errors for each layer from backpropagation
     */
    public ArrayList<Matrix> backPropagate(Matrix expected) {
        ArrayList<Matrix> errors = new ArrayList<>();
        for (int layer = layerCount - 1; layer > 0; layer--) {
            if (layer == layerCount - 1) {
                errors.add(layers.get(layer).getErrorsExpected(expected));
                errors.add(layers.get(layer).getErrors(errors.get(errors.size() - 1)));
            } else {
                errors.add(layers.get(layer).getErrors(errors.get(errors.size() - 1)));
            }
        }
        return errors;
    }

    /**
     * Get the loss.
     *
     * @param loss     loss function to use
     * @param output   actual output of model
     * @param expected expected output
     * @return loss of model
     */
    public double getLoss(Loss loss, Matrix output, Matrix expected) {
        return loss.getLoss(output, expected);
    }

    /**
     * Get the loss.
     *
     * @param loss     loss function to use
     * @param output   actual output of model
     * @param expected expected output
     * @return loss of model
     */
    public double getLoss(Loss loss, ArrayList<Matrix> output, ArrayList<Matrix> expected) {
        return loss.getLoss(output, expected);
    }

    /**
     * Get a metric of the model.
     *
     * @param metric   metric function to use
     * @param output   actual output of model
     * @param expected expected output
     * @return calculated metric
     */
    public double getMetric(Metric metric, Matrix output, Matrix expected) {
        return metric.getMetric(output, expected);
    }

    /**
     * Get a metric of the model.
     *
     * @param metric   metric function to use
     * @param output   actual output of model
     * @param expected expected output
     * @return calculated metric
     */
    public double getMetric(Metric metric, ArrayList<Matrix> output, ArrayList<Matrix> expected) {
        return metric.getMetric(output, expected);
    }

    /**
     * Update the model after backpropagation.
     *
     * @param errors       errors obtained from backpropagation
     * @param learningRate learning rate of updating
     */
    private void update(ArrayList<Matrix> errors, double learningRate) {
        for (int layer = 1; layer < layerCount; layer++) {
            int eLayer = layerCount - layer - 1;
            layers.get(layer).update(errors.get(eLayer), learningRate);
        }
    }

    /**
     * Save the current model to a file.
     * <p>
     * Note: the model can continue to be trained and used to predict after saving.
     * </p>
     *
     * @param filePath path to file to save to
     */
    public void saveToFile(String filePath) {
        try {
            FileOutputStream fos = new FileOutputStream(filePath);
            GZIPOutputStream gzipOut = new GZIPOutputStream(fos);
            ObjectOutputStream oos = new ObjectOutputStream(gzipOut);

            oos.writeObject(this);

            oos.flush();
            oos.close();
            gzipOut.finish();
            fos.flush();
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    protected static class FitPrint extends Thread {
        protected String fitOutput = "";
        protected boolean stopped = false;

        public void setOutput(String fitOutput) {
            this.fitOutput = fitOutput;
        }

        public void stopThread() {
            stopped = true;
        }

        public void run() {
            while (!stopped) {
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.print(fitOutput);
            }
        }
    }

    public static class FitBuilder {
        protected Matrix input, expected;
        protected double learningRate;
        protected int batchSize = 1;
        protected int epochs = 1;
        protected ArrayList<Metric> metrics = new ArrayList<>();
        protected boolean hasEval = false;
        protected Matrix evalInput, evalExpected;
        protected int logInterval = 1;

        public FitBuilder(Matrix input, Matrix expected) {
            this.input = input;
            this.expected = expected;
        }

        public FitBuilder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public FitBuilder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public FitBuilder epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }

        public FitBuilder metrics(ArrayList<Metric> metrics) {
            this.metrics = metrics;
            return this;
        }

        public FitBuilder eval(Matrix evalInput, Matrix evalExpected) {
            this.evalInput = evalInput;
            this.evalExpected = evalExpected;
            hasEval = true;
            return this;
        }

        public FitBuilder logInterval(int logInterval) {
            this.logInterval = logInterval;
            return this;
        }
    }
}
