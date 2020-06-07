package me.yixqiao.jlearn.genetic;

import me.yixqiao.jlearn.exceptions.NeuralNetworkException;
import me.yixqiao.jlearn.layers.InputLayer;
import me.yixqiao.jlearn.layers.Layer;
import me.yixqiao.jlearn.matrix.Matrix;

import java.util.ArrayList;

/**
 * Population of individuals.
 */
public abstract class Population {
    /**
     * Number of individuals.
     */
    protected final int indivCount;
    /**
     * ArrayList of layers as a template for individuals.
     */
    protected final ArrayList<Layer> layers;
    /**
     * Which generation the population is on.
     */
    public int generation = 0;
    /**
     * Array of individuals.
     */
    protected Individual[] individuals;
    /**
     * Number of layers.
     */
    protected int layerCount;

    /**
     * Create a new population.
     *
     * @param indivCount number of individuals
     */
    public Population(int indivCount) {
        this.indivCount = indivCount;
        layers = new ArrayList<>();
        individuals = new Individual[indivCount];
    }

    /**
     * Add a layer to the general layer template.
     *
     * @param layer the layer
     * @return the population itself to allow for daisy chaining
     */
    public Population addLayer(Layer layer) {
        layers.add(layer);
        return this;
    }

    /**
     * Initialize all layers.
     * <p>
     * This function should not be called directly; the {@link #init() init} method does this already.
     * </p>
     */
    protected void initLayers() {
        layerCount = layers.size();
        if (!(layers.get(0) instanceof InputLayer)) {
            throw new NeuralNetworkException("First layer is not an input layer");
        }
        for (int layer = 0; layer < layerCount - 1; layer++) {
            layers.get(layer + 1).initLayer(layers.get(layer).getOutSize(), layers.get(layer).getActivation());
        }
    }

    /**
     * Initialize the population.
     * <p>
     * This must be run after adding all layers and before simulating.
     * </p>
     */
    public void init() {
        initLayers();
        for (int i = 0; i < indivCount; i++) {
            ArrayList<Layer> iLayers = new ArrayList<>();
            for (Layer l : layers)
                iLayers.add(l.cloneSettings());
            individuals[i] = new Individual(iLayers);
        }
    }

    /**
     * Forward propagate for a single individual.
     *
     * @param indivNum the individual
     * @param x        the input matrix
     * @return the output
     */
    public Matrix forwardPropagate(int indivNum, Matrix x) {
        return individuals[indivNum].forwardPropagate(x);
    }

    /**
     * Simulate generations.
     *
     * @param generations number of generations
     */
    public void simGenerations(int generations) {
        for (int g = 0; g < generations; g++) {
            oneGeneration();
        }
    }

    /**
     * Simulate one generation.
     */
    public void oneGeneration() {
        resetScores();
        calcScores();
        select();
        generation++;
    }

    /**
     * Reset scores. This is called within {@link #oneGeneration() oneGeneration}.
     */
    protected void resetScores() {
        for (int i = 0; i < indivCount; i++) {
            individuals[i].score = 0;
        }
    }

    /**
     * Calculate the scores for each network.
     */
    protected abstract void calcScores();

    /**
     * Select and randomize networks for the next generation.
     */
    protected abstract void select();

    /**
     * Print scores of the best networks.
     *
     * @param n number of networks to print
     */
    protected void printBest(int n) {
        for (int i = 0; i < n; i++) {
            System.out.printf("%.5f", individuals[i].score);
            if (i < n - 1)
                System.out.print(", ");
            // forwardPropagate(i, input).printMatrix();
            // System.out.println();
        }
    }

}
