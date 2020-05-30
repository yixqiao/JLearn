package me.yixqiao.jlearn.testing;

import me.yixqiao.jlearn.layers.Dense;
import me.yixqiao.jlearn.metrics.Accuracy;
import me.yixqiao.jlearn.metrics.Metric;
import me.yixqiao.jlearn.models.Model;

import java.io.*;
import java.util.ArrayList;

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
            ObjectInputStream ois = new ObjectInputStream((new FileInputStream(("m.tmp"))));
            model = (Model) ois.readObject();
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
        model.fit(inputs, outputs, evalInputs, evalOutputs, 0.01, 4, 10, 1, metrics);

        printPredictions();

        // TODO: https://stackoverflow.com/questions/5934495/implementing-in-memory-compression-for-objects-in-java
        try {
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("m.tmp"));
            oos.writeObject(model);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
