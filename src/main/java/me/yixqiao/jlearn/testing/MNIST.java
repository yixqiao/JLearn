package me.yixqiao.jlearn.testing;

import jdk.jshell.spi.ExecutionControl;
import me.yixqiao.jlearn.activations.*;
import me.yixqiao.jlearn.datasets.DatasetTT;
import me.yixqiao.jlearn.datasets.MNISTDigits;
import me.yixqiao.jlearn.losses.MeanSquaredError;
import me.yixqiao.jlearn.matrix.Matrix;
import me.yixqiao.jlearn.layers.Dense;
import me.yixqiao.jlearn.layers.InputLayer;
import me.yixqiao.jlearn.losses.CrossEntropy;
import me.yixqiao.jlearn.metrics.Accuracy;
import me.yixqiao.jlearn.metrics.Metric;
import me.yixqiao.jlearn.models.Model;
import me.yixqiao.jlearn.settings.Settings;

import java.io.*;
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
    ArrayList<Matrix> inputsALC = new ArrayList<>();
    ArrayList<Matrix> outputsALC = new ArrayList<>();
    Model model;

    public static void main(String[] args) {
        Settings.THREAD_COUNT /= 2; // Use physical core count
        MNIST mnist = new MNIST();
        // mnist.writeDataset();
        mnist.initInputs();
        mnist.buildModel();
        mnist.train();
    }

    protected void buildModel() {
        model = new Model();
        model.addLayer(new InputLayer(28 * 28))
                .addLayer(new Dense(64, new ReLU()))
                .addLayer(new Dense(32, new ReLU()))
                .addLayer(new Dense(10, new Softmax()));

        model.buildModel(new CrossEntropy());

        model.printSummary();
    }

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

    protected void evaluateModel() {
        ArrayList<Metric> metrics = new ArrayList<Metric>() {{
            add(new Accuracy());
        }};
        System.out.print("Eval: ");
        model.evaluate(evalInputs, evalOutputs, metrics);
    }

    protected void afterTrain() {
        // Override in subclass
    }

    protected void printPredictions() {
        for (int i = 0; i < inputsALC.size(); i++) {
            Matrix output = model.predict(inputsALC.get(i));
            for (int j = 0; j < output.cols; j++) {
                System.out.print(String.format("%.3f", output.mat[0][j]));
                if (j != output.cols - 1) System.out.print("\t");
            }

            System.out.print("\t-\t");
            for (int j = 0; j < outputs.cols; j++) {
                System.out.print(String.format("%.3f", outputsALC.get(i).mat[0][j]));
                if (j != outputs.cols - 1) System.out.print("\t");
            }

            System.out.println();
        }
    }

    protected void writeDataset() {
        // Flattens all images
        try {
            BufferedReader br = new BufferedReader(new FileReader("datasets/mnist/csv/mnist_train.csv"));
            DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream("datasets/mnist/data/train.dat")));
            String line;
            br.readLine(); // Discard first line
            for (int imgCount = 0; imgCount < 60000; imgCount++) {
                line = br.readLine();
                String[] values = line.split(",");
                Matrix output = new Matrix(1, 10);

                output.mat[0][Integer.parseInt(values[0])] = 1;
                dos.writeByte((byte) Integer.parseInt(values[0]));

                Matrix input = new Matrix(1, 28 * 28);
                for (int i = 0; i < 28 * 28; i++) {
                    input.mat[0][i] = Double.parseDouble(values[1 + i]);
                    dos.writeByte((byte) (input.mat[0][i] - 128));
                }

                input.multiplyIP(1.0 / 255);
            }
            dos.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            BufferedReader br = new BufferedReader(new FileReader("datasets/mnist/csv/mnist_test.csv"));
            DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream("datasets/mnist/data/test.dat")));
            String line;
            br.readLine(); // Discard first line
            for (int imgCount = 0; imgCount < 10000; imgCount++) {
                line = br.readLine();
                String[] values = line.split(",");
                Matrix output = new Matrix(1, 10);

                output.mat[0][Integer.parseInt(values[0])] = 1;
                dos.writeByte((byte) Integer.parseInt(values[0]));

                Matrix input = new Matrix(1, 28 * 28);
                for (int i = 0; i < 28 * 28; i++) {
                    input.mat[0][i] = Double.parseDouble(values[1 + i]);
                    dos.writeByte((byte) (input.mat[0][i] - 128));
                }

                input.multiplyIP(1.0 / 255);
            }
            dos.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Finished writing to file.");

        System.exit(0);
    }

    protected void initInputs() {
        DatasetTT data = MNISTDigits.load();
        inputs = data.train.x;
        outputs = data.train.y;
        evalInputs = data.test.x;
        evalOutputs = data.test.y;
    }

    protected void initInputsOld() {
        ArrayList<Matrix> inputsAL = new ArrayList<>();
        ArrayList<Matrix> outputsAL = new ArrayList<>();

        // Load training
        try {
            DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream("datasets/mnist/data/train.dat")));
            for (int i = 0; i < 60000; i++) {
                Matrix output = new Matrix(1, 10);
                output.mat[0][dis.readByte()] = 1;

                Matrix input = new Matrix(1, 28 * 28);
                for (int j = 0; j < 28 * 28; j++) {
                    input.mat[0][j] = dis.readByte() + 128;
                }

                input.multiplyIP(1.0 / 255);
                inputsAL.add(input);
                outputsAL.add(output);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        inputs = new Matrix(inputsAL.size(), inputsAL.get(0).cols);
        outputs = new Matrix(outputsAL.size(), outputsAL.get(0).cols);

        for (int i = 0; i < inputsAL.size(); i++) {
            for (int j = 0; j < inputsAL.get(i).cols; j++) {
                inputs.mat[i][j] = inputsAL.get(i).mat[0][j];
            }
            for (int j = 0; j < outputsAL.get(i).cols; j++) {
                outputs.mat[i][j] = outputsAL.get(i).mat[0][j];
            }
        }

        inputsAL = new ArrayList<>();
        outputsAL = new ArrayList<>();

        // Load testing
        try {
            DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream("datasets/mnist/data/test.dat")));
            for (int i = 0; i < 10000; i++) {
                Matrix output = new Matrix(1, 10);
                output.mat[0][dis.readByte()] = 1;

                Matrix input = new Matrix(1, 28 * 28);
                for (int j = 0; j < 28 * 28; j++) {
                    input.mat[0][j] = dis.readByte() + 128;
                }

                input.multiplyIP(1.0 / 255);
                inputsAL.add(input);
                outputsAL.add(output);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        evalInputs = new Matrix(inputsAL.size(), inputsAL.get(0).cols);
        evalOutputs = new Matrix(outputsAL.size(), outputsAL.get(0).cols);

        for (int i = 0; i < inputsAL.size(); i++) {
            for (int j = 0; j < inputsAL.get(i).cols; j++) {
                evalInputs.mat[i][j] = inputsAL.get(i).mat[0][j];
            }
            for (int j = 0; j < outputsAL.get(i).cols; j++) {
                evalOutputs.mat[i][j] = outputsAL.get(i).mat[0][j];
            }
        }

        System.out.println("Finished reading inputs from file.");

        for (int i = 0; i < inputsAL.size(); i += (10000 / 10)) {
            inputsALC.add(inputsAL.get(i));
            outputsALC.add(outputsAL.get(i));
        }
    }
}

