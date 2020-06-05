package me.yixqiao.jlearn.testing;

import me.yixqiao.jlearn.activations.ReLU;
import me.yixqiao.jlearn.activations.Sigmoid;
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
import java.util.Random;

public class Genetic {
    static Random random = new Random();
    static Matrix input = new Matrix(new double[][]{{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}});


    public static void main(String[] args) {
        Population p = new Pop(1000)
                .addLayer(new InputLayer(10))
                .addLayer(new Dense(10, new Sigmoid()));

        p.init();


        // for (int i = 0; i < 10; i++)
        //     p.forwardPropagate(i, input).printMatrix();

        for (int i = 0; i < 1000; i++) {
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
                Matrix output = forwardPropagate(i, input);
                // output.printMatrix();
                individuals[i].score = new MeanSquaredError().getLoss(output, good);
            }
        }

        @Override
        protected void select() {
            Arrays.sort(individuals, new SortIndivs());
            for (int i = 0; i < 10; i++) {
                System.out.print(individuals[i].score + "\t: ");
                forwardPropagate(i, new Matrix(new double[][]{{1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1}})).printMatrix();
                System.out.println();
            }

            for (int i = 0; i < indivCount / 2; i++) {
                individuals[indivCount / 2 + i] = individuals[i].cloneIndividual();
                // individuals[i + indivCount / 2] = new Individual(layers);
                randomize(individuals[indivCount / 2 + i]);
            }

            for (int i = 0; i < indivCount; i++) {
                individuals[i].score = 0;
            }
        }

        private void randomize(Individual ind) {
            for (Layer l : ind.layers) {
                if (l instanceof Dense) {
                    ((Dense) l).weights.applyEachIP(x -> x + random.nextGaussian() * 0.05 * ((random.nextDouble() < 0.2) ? 1 : 0));
                    ((Dense) l).biases.applyEachIP(x -> x + random.nextGaussian() * 0.05 * ((random.nextDouble() < 0.2) ? 1 : 0));
                }
            }
        }

        protected static class SortIndivs implements Comparator<Individual> {
            public int compare(Individual a, Individual b) {
                return Double.compare(a.score, b.score);
            }
        }
    }
}
