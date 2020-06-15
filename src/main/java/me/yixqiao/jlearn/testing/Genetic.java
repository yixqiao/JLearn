package me.yixqiao.jlearn.testing;

import me.yixqiao.jlearn.activations.Sigmoid;
import me.yixqiao.jlearn.genetic.Individual;
import me.yixqiao.jlearn.genetic.Population;
import me.yixqiao.jlearn.layers.Dense;
import me.yixqiao.jlearn.layers.InputLayer;
import me.yixqiao.jlearn.losses.MeanSquaredError;
import me.yixqiao.jlearn.matrix.Matrix;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

/**
 * Run a genetic algorithm.
 */
public class Genetic {
    static Random random = new Random();
    static Matrix input = new Matrix(new double[][]{{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}});


    /**
     * Run.
     *
     * @param args args
     */
    public static void main(String[] args) {
        Pop p = (Pop) new Pop(1000)
                .addLayer(new InputLayer(10))
                .addLayer(new Dense(10, new Sigmoid()));

        p.init();

        p.simGenerations(200);
    }

    private static class Pop extends Population {
        public Pop(int indivCount) {
            super(indivCount);
        }

        @Override
        protected void calcScores() {
            Matrix good = new Matrix(new double[][]{{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}});
            for (int i = 0; i < indivCount; i++) {
                Matrix output = forwardPropagate(i, input);
                individuals[i].score = new MeanSquaredError().getLoss(output, good);
            }
        }

        @Override
        public void simGenerations(int generations) {
            super.simGenerations(generations);
            done();
        }

        @Override
        protected void select() {
            Arrays.sort(individuals, new SortIndivs());

            System.out.print(generation + ": ");
            printBest(3);
            System.out.println();

            for (int i = 0; i < indivCount / 2; i++) {
                individuals[indivCount / 2 + i] = individuals[i].cloneIndividual();
                randomize(individuals[indivCount / 2 + i]);
            }
        }

        private void randomize(Individual ind) {
            ind.applyWeightsIP(x -> (random.nextDouble() < 0.1) ? x * (1 + random.nextGaussian() * 0.02) : x);
            ind.applyBiasesIP(x -> (random.nextDouble() < 0.1) ? x * (1 + random.nextGaussian() * 0.02) : x);
            ind.applyWeightsIP(x -> (random.nextDouble() < 0.2) ? x + random.nextGaussian() * 0.05 : x);
            ind.applyBiasesIP(x -> (random.nextDouble() < 0.2) ? x + random.nextGaussian() * 0.05 : x);
        }

        /**
         * Method called when finished simulating.
         */
        public void done() {
            forwardPropagate(0, input).printMatrix();
        }

        protected static class SortIndivs implements Comparator<Individual> {
            public int compare(Individual a, Individual b) {
                return Double.compare(a.score, b.score);
            }
        }
    }
}
