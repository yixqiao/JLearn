package me.yixqiao.jlearn.testing;

import me.yixqiao.jlearn.activations.LeakyReLU;
import me.yixqiao.jlearn.layers.Dense;
import me.yixqiao.jlearn.layers.Layer;
import me.yixqiao.jlearn.matrix.Matrix;
import me.yixqiao.jlearn.models.Model;

import java.io.*;

public class SerializeTest {
    public static void main(String[] args) {
        // layer();
        // matrix();
        System.out.println("Nothing here.");
    }

    private static void layer() {
        Dense l = new Dense(12, new LeakyReLU(0.12345));
        l.initLayer(45, new LeakyReLU(0.6789));
        try {
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("m.tmp"));
            oos.writeObject(l);
        } catch (IOException e) {
            e.printStackTrace();
        }

        Dense l2 = null;
        try {
            ObjectInputStream ois = new ObjectInputStream((new FileInputStream(("m.tmp"))));
            l2 = (Dense) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        System.out.println(l2.getOutSize());
        System.exit(0);
    }

    private static void matrix() {
        Matrix m = new Matrix(new double[][]{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});
        m.printMatrix();
        try {
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("m.tmp"));
            oos.writeObject(m);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Matrix m2 = null;
        try {
            ObjectInputStream ois = new ObjectInputStream((new FileInputStream(("m.tmp"))));
            m2 = (Matrix) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        m2.printMatrix();
        System.exit(0);
    }
}
