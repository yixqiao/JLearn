package me.yixqiao.jlearn.testing;

import me.yixqiao.jlearn.activations.ReLU;
import me.yixqiao.jlearn.genetic.Individual;
import me.yixqiao.jlearn.genetic.Population;
import me.yixqiao.jlearn.layers.Dense;
import me.yixqiao.jlearn.layers.InputLayer;
import me.yixqiao.jlearn.layers.Layer;
import me.yixqiao.jlearn.losses.MeanSquaredError;
import me.yixqiao.jlearn.matrix.Matrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

public class Genetic {
    public static void main(String[] args) {
        Population p = new Pop(10)
                .addLayer(new InputLayer(10))
                .addLayer(new Dense(36, new ReLU()))
                .addLayer(new Dense(10, new ReLU()));

        p.init();

        Matrix input = new Matrix(new double[][]{{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}});

        // for (int i = 0; i < 10; i++)
        //     p.forwardPropagate(i, input).printMatrix();

        for (int i = 0; i < 10000; i++) {
            System.out.println(p.generation);
            p.oneGeneration();
        }
    }

    private static class Pop extends Population {
        public Pop(int indivCount) {
            super(indivCount);
        }

        @Override
        protected void calcScores() {
            // Matrix good = new Matrix(new double[][]{{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}});
            // Matrix good = new Matrix(new double[][]{{0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1, 0}});
            Matrix good = new Matrix(new double[][]{{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}});
            for (int i = 0; i < indivCount; i++) {
                Matrix output = forwardPropagate(i, new Matrix(new double[][]{{1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1}}));
                // output.printMatrix();
                individuals[i].score = new MeanSquaredError().getLoss(output, good);
            }
        }

        @Override
        protected void select() {
            Arrays.sort(individuals, new SortIndivs());
            for (int i = 0; i < indivCount; i++) {
                System.out.print(individuals[i].score + "\t: ");
                forwardPropagate(i, new Matrix(1, 10, new Matrix.Init.Gaussian(0.5))).printMatrix();
            }

            for (int i = 0; i < indivCount / 2; i++) {
                ArrayList<Layer> iLayers = new ArrayList<>();
                for (Layer l : layers)
                    iLayers.add(l.cloneSettings());
                individuals[i + indivCount / 2] = new Individual(iLayers);
            }
        }

        protected static class SortIndivs implements Comparator<Individual> {
            public int compare(Individual a, Individual b) {
                return Double.compare(a.score, b.score);
            }
        }
    }
}
