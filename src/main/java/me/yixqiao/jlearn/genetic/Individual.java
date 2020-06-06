package me.yixqiao.jlearn.genetic;

import me.yixqiao.jlearn.layers.Dense;
import me.yixqiao.jlearn.layers.Layer;
import me.yixqiao.jlearn.matrix.Matrix;

import java.util.ArrayList;
import java.util.function.ToDoubleFunction;

/**
 * Individual neural network.
 */
public class Individual {
    /**
     * Layers.
     */
    public final ArrayList<Layer> layers;
    /**
     * Score (used for selecting).
     */
    public double score;
    /**
     * Number of layers (including input).
     */
    protected int layerCount;

    /**
     * Create a new individual.
     *
     * @param layers layers from population
     */
    public Individual(ArrayList<Layer> layers) {
        this.layers = layers;
        layerCount = layers.size();
    }

    /**
     * Forward propagate.
     *
     * @param x input matrix
     * @return output
     */
    public Matrix forwardPropagate(Matrix x) {
        Matrix activations = x.clone();
        for (int layerNum = 0; layerNum < layerCount; layerNum++) {
            activations = layers.get(layerNum).forwardPropagate(activations);
        }
        return activations;
    }

    /**
     * Apply to each weight.
     *
     * @param function a function
     */
    public void applyWeightsIP(ToDoubleFunction<Double> function) {
        for (Layer l : layers)
            if (l instanceof Dense)
                ((Dense) l).weights.applyEachIP(function);
    }

    /**
     * Apply to each bias.
     *
     * @param function a function
     */
    public void applyBiasesIP(ToDoubleFunction<Double> function) {
        for (Layer l : layers)
            if (l instanceof Dense)
                ((Dense) l).biases.applyEachIP(function);
    }

    /**
     * Clone the individual.
     *
     * @return the clone
     */
    public Individual cloneIndividual() {
        ArrayList<Layer> newLayers = new ArrayList<>();
        for (Layer l : layers)
            newLayers.add(l.cloneLayer());
        return new Individual(newLayers);
    }
}
