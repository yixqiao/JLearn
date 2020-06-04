package me.yixqiao.jlearn.testing;

import me.yixqiao.jlearn.activations.ReLU;
import me.yixqiao.jlearn.genetic.Population;
import me.yixqiao.jlearn.layers.Dense;
import me.yixqiao.jlearn.layers.InputLayer;
import me.yixqiao.jlearn.matrix.Matrix;

public class Genetic {
    public static void main(String[] args) {
        Population p = new Population(10)
                .addLayer(new InputLayer(10))
                .addLayer(new Dense(36, new ReLU()))
                .addLayer(new Dense(10, new ReLU()));
        p.initLayers();
        p.initPop();

        Matrix input = new Matrix(new double[][]{{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}});

        for (int i = 0; i < 10; i++)
            p.predict(i, input).printMatrix();
    }
}
