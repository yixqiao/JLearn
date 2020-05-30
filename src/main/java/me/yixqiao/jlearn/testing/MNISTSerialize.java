package me.yixqiao.jlearn.testing;

import me.yixqiao.jlearn.layers.Dense;
import me.yixqiao.jlearn.metrics.Accuracy;
import me.yixqiao.jlearn.metrics.Metric;
import me.yixqiao.jlearn.models.Model;

import java.io.*;
import java.util.ArrayList;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class MNISTSerialize extends MNIST {
    public static void main(String[] args) {
        runLoad();
    }

    protected static void runTrain() {
        MNISTSerialize mnist = new MNISTSerialize();
        // mnist.writeDataset();
        mnist.initInputs();
        mnist.buildModel();
        mnist.train();
    }

    protected static void runLoad() {
        MNISTSerialize mnist = new MNISTSerialize();
        mnist.loadModel();
        mnist.initInputs();
        mnist.printPredictions();
    }

    protected void loadModel() {
        Model model = null;
        try {
            FileInputStream fis = new FileInputStream("m.tmp");
            GZIPInputStream gzipIn = new GZIPInputStream(fis);
            ObjectInputStream ois = new ObjectInputStream(gzipIn);

            model = (Model) ois.readObject();

            fis.close();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        this.model = model;
        System.out.println("Done.");
    }

    @Override
    protected void train() {
        printPredictions();

        ArrayList<Metric> metrics = new ArrayList<>() {{
            add(new Accuracy());
        }};
        model.fit(inputs, outputs, evalInputs, evalOutputs, 0.01, 4, 2, 1, metrics);

        printPredictions();

        try {
            FileOutputStream fos = new FileOutputStream("m.tmp");
            GZIPOutputStream gzipOut = new GZIPOutputStream(fos);
            ObjectOutputStream oos = new ObjectOutputStream(gzipOut);

            oos.writeObject(model);

            oos.flush();
            oos.close();
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
