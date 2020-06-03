package me.yixqiao.jlearn.models;

import me.yixqiao.jlearn.exceptions.NeuralNetworkException;
import me.yixqiao.jlearn.layers.InputLayer;
import me.yixqiao.jlearn.matrix.Matrix;
import me.yixqiao.jlearn.layers.Layer;
import me.yixqiao.jlearn.losses.Loss;
import me.yixqiao.jlearn.metrics.Metric;

import java.io.*;
import java.util.ArrayList;
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
     * @param fb FitBuilder instance
     */
    public void fit(FitBuilder fb) {
        Matrix trainX = fb.trainX;
        Matrix trainY = fb.trainY;
        double learningRate = fb.learningRate;
        ArrayList<Metric> metrics = fb.metrics;
        int batchSize = fb.batchSize;
        int epochs = fb.epochs;
        int logInterval = fb.logInterval;

        int totalSamples = trainX.rows;
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
                Matrix batchX = new Matrix(batchSize, trainX.cols);
                Matrix batchY = new Matrix(batchSize, trainY.cols);
                for (int i = 0; i < batchSize; i++) {
                    for (int j = 0; j < trainX.cols; j++) {
                        batchX.mat[i][j] = trainX.mat[indices.get(batchNum * batchSize + i)][j];
                    }
                    for (int j = 0; j < trainY.cols; j++) {
                        batchY.mat[i][j] = trainY.mat[indices.get(batchNum * batchSize + i)][j];
                    }
                }

                Matrix batchOut = forwardPropagate(batchX);

                if ((epoch + 1) % logInterval == 0) {
                    lossA += getLoss(loss, batchOut, batchY);
                    for (int i = 0; i < metrics.size(); i++)
                        metricA[i] += getMetric(metrics.get(i), batchOut, batchY);
                }

                errors = backPropagate(batchY);
                update(errors, learningRate);

                if ((epoch + 1) % logInterval == 0) {
                    String fitOutput = "";

                    double timeElapsed = (double) (System.nanoTime() - epochStart) / 1e9;
                    fitOutput += String.format("\rE: %d - T: %5.2fs - L: %.5f", epoch + 1, timeElapsed, lossA / (batchNum + 1));

                    for (int i = 0; i < metrics.size(); i++)
                        fitOutput += String.format((" - " + metrics.get(i).getFormatString()), metricA[i] / (batchNum + 1));

                    fitOutput += " - [";
                    int progress = (int) ((double) batchNum / ((double) totalSamples / batchSize) * 20);
                    for (int i = 0; i < progress; i++)
                        fitOutput += "=";
                    fitOutput += ">";
                    for (int i = 0; i < 20 - progress - 1; i++)
                        fitOutput += ".";
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

                if (fb.hasEval)
                    toPrint += " - Evaluating...";

                for (int i = toPrint.length(); i < maxLineLen; i++)
                    toPrint += " ";

                System.out.print(toPrint);

                lossA /= (double) (totalSamples / batchSize);
                for (int i = 0; i < metrics.size(); i++)
                    metricA[i] /= (double) (totalSamples / batchSize);

                Matrix evalY = null;
                if (fb.hasEval)
                    evalY = forwardPropagate(fb.evalX);

                double timeElapsed = (double) (System.nanoTime() - epochStart) / 1e9;

                String finalOutput = String.format("\rE: %d - T: %5.2fs - L: %.5f", epoch + 1, timeElapsed, lossA);

                for (int i = 0; i < metrics.size(); i++)
                    finalOutput += String.format((" - " + metrics.get(i).getFormatString()), metricA[i]);

                if (fb.hasEval) {
                    double evalLoss = getLoss(loss, evalY, fb.evalY);
                    finalOutput += String.format(" - EL: %.5f", evalLoss);

                    for (Metric metric : metrics)
                        finalOutput += String.format((" - E" + metric.getFormatString()),
                                getMetric(metric, evalY, fb.evalY));
                }

                for (int i = 0; i < 3; i++) {
                    System.out.print(finalOutput);
                    System.out.flush();
                }

                System.out.println();
            }

            onEpochEnd(epoch);
        }
    }

    /**
     * Train on a single batch of input and output.
     *
     * @param x            input data to train on
     * @param y            expected outputs
     * @param learningRate learning rate of training
     */
    public void trainOnBatch(Matrix x, Matrix y, double learningRate) {
        forwardPropagate(x);
        ArrayList<Matrix> errors = backPropagate(y);
        update(errors, learningRate);
    }

    /**
     * Predict on a batch of input.
     *
     * @param x input data to feed forward
     * @return prediction of model for input
     */
    public Matrix predict(Matrix x) {
        return forwardPropagate(x);
    }

    public void comparePredictions(Matrix input, Matrix output, int printNum) {
        Matrix x = new Matrix(printNum, input.cols);
        int skip = input.rows / printNum;

        for (int i = 0; i < printNum; i++) {
            for (int j = 0; j < input.cols; j++) {
                x.mat[i][j] = input.mat[i * skip][j];
            }
        }

        Matrix y = forwardPropagate(x);

        for (int i = 0; i < printNum; i++) {
            for (int j = 0; j < y.cols; j++) {
                System.out.print(String.format("%.3f", y.mat[i][j]));
                if (j != y.cols - 1) System.out.print("\t");
            }

            System.out.print("\t-\t");
            for (int j = 0; j < output.cols; j++) {
                System.out.print(String.format("%.3f", output.mat[i][j]));
                if (j != output.cols - 1) System.out.print("\t");
            }

            System.out.println();
        }
    }

    /**
     * Evaluate the performance of the model.
     *
     * @param x       input data
     * @param y       expected outputs
     * @param metrics metrics to display
     */
    public void evaluate(Matrix x, Matrix y, ArrayList<Metric> metrics) {
        Matrix output = forwardPropagate(x);

        double evalLoss = getLoss(loss, output, y);
        System.out.printf("L: %.5f", evalLoss);

        for (int i = 0; i < metrics.size(); i++)
            System.out.printf((", " + metrics.get(i).getFormatString()),
                    getMetric(metrics.get(i), output, y));

        System.out.println();
    }

    /**
     * Forward propagate a batch of input.
     *
     * @param x input data to feed forward
     * @return result of model for input
     */
    public Matrix forwardPropagate(Matrix x) {
        Matrix activations = x.clone();
        for (int layerNum = 0; layerNum < layerCount; layerNum++) {
            activations = layers.get(layerNum).forwardPropagate(activations);
        }
        return activations;
    }

    /**
     * Backpropagate model after forward propagating input.
     *
     * @param y expected output for model
     * @return errors for each layer from backpropagation
     */
    public ArrayList<Matrix> backPropagate(Matrix y) {
        ArrayList<Matrix> errors = new ArrayList<>();
        for (int layer = layerCount - 1; layer > 0; layer--) {
            if (layer == layerCount - 1) {
                errors.add(layers.get(layer).getErrorsExpected(y));
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
     * @param loss loss function to use
     * @param out  actual output of model
     * @param y    expected output
     * @return loss of model
     */
    public double getLoss(Loss loss, Matrix out, Matrix y) {
        return loss.getLoss(out, y);
    }

    /**
     * Get the loss.
     *
     * @param loss loss function to use
     * @param out  actual output of model
     * @param y    expected output
     * @return loss of model
     */
    public double getLoss(Loss loss, ArrayList<Matrix> out, ArrayList<Matrix> y) {
        return loss.getLoss(out, y);
    }

    /**
     * Get a metric of the model.
     *
     * @param metric metric function to use
     * @param out    actual output of model
     * @param y      expected output
     * @return calculated metric
     */
    public double getMetric(Metric metric, Matrix out, Matrix y) {
        return metric.getMetric(out, y);
    }

    /**
     * Get a metric of the model.
     *
     * @param metric metric function to use
     * @param out    actual output of model
     * @param y      expected output
     * @return calculated metric
     */
    public double getMetric(Metric metric, ArrayList<Matrix> out, ArrayList<Matrix> y) {
        return metric.getMetric(out, y);
    }

    /**
     * Update the model after backpropagation.
     *
     * @param errors       errors obtained from backpropagation
     * @param learningRate learning rate of updating
     */
    protected void update(ArrayList<Matrix> errors, double learningRate) {
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

    /**
     * Method that is called at the end of every epoch.
     * <p>
     * Subclasses of Model can easily override this method to do something once each epoch ends.
     * </p>
     *
     * @param epoch epoch number (0-indexed)
     */
    protected void onEpochEnd(int epoch) {
        // Do anything here
    }

    /**
     * Thread to print progress when fitting.
     */
    protected static class FitPrint extends Thread {
        /**
         * Output to print.
         */
        protected String fitOutput = "";
        /**
         * Whether to stop the thread.
         */
        protected boolean stopped = false;

        /**
         * Set output to print.
         *
         * @param fitOutput output
         */
        public void setOutput(String fitOutput) {
            this.fitOutput = fitOutput;
        }

        /**
         * Stop the printing.
         */
        public void stopThread() {
            stopped = true;
        }

        /**
         * Run the thread.
         */
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

    /**
     * Builder class for fit operation.
     */
    public static class FitBuilder {
        /**
         * Training input.
         */
        protected Matrix trainX;
        /**
         * Training correct values.
         */
        protected Matrix trainY;
        /**
         * Learning rate.
         */
        protected double learningRate;
        /**
         * Batch size.
         */
        protected int batchSize = 1;
        /**
         * Number of epochs.
         */
        protected int epochs = 1;
        /**
         * Metrics to gauge model performance.
         */
        protected ArrayList<Metric> metrics = new ArrayList<>();
        /**
         * Whether to use an evaluation dataset.
         */
        protected boolean hasEval = false;
        /**
         * Evaluation input.
         */
        protected Matrix evalX;
        /**
         * Evaluation correct values.
         */
        protected Matrix evalY;
        /**
         * Log to console every n epochs.
         */
        protected int logInterval = 1;

        /**
         * Create a new FitBuilder.
         *
         * @param trainX training input
         * @param trainY training correct values
         */
        public FitBuilder(Matrix trainX, Matrix trainY) {
            this.trainX = trainX;
            this.trainY = trainY;
        }

        /**
         * Set learning rate.
         *
         * @param learningRate learning rate
         * @return the instance for daisy chaining
         */
        public FitBuilder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        /**
         * Set batch size.
         *
         * @param batchSize batch size
         * @return the instance for daisy chaining
         */
        public FitBuilder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }


        /**
         * Set the epoch number.
         *
         * @param epochs number of epochs
         * @return the instance for daisy chaining
         */
        public FitBuilder epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }

        /**
         * Set metrics.
         *
         * @param metrics list of metrics
         * @return the instance for daisy chaining
         */
        public FitBuilder metrics(ArrayList<Metric> metrics) {
            this.metrics = metrics;
            return this;
        }

        /**
         * Set evaluation dataset.
         *
         * @param evalX evaluation input
         * @param evalY evaluation correct values
         * @return the instance for daisy chaining
         */
        public FitBuilder eval(Matrix evalX, Matrix evalY) {
            this.evalX = evalX;
            this.evalY = evalY;
            hasEval = true;
            return this;
        }

        /**
         * Set the log interval.
         *
         * @param logInterval interval to log
         * @return the instance for daisy chaining
         */
        public FitBuilder logInterval(int logInterval) {
            this.logInterval = logInterval;
            return this;
        }
    }
}
