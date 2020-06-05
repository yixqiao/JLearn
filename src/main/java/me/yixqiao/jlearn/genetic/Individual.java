package me.yixqiao.jlearn.genetic;

import me.yixqiao.jlearn.exceptions.NeuralNetworkException;
import me.yixqiao.jlearn.layers.InputLayer;
import me.yixqiao.jlearn.layers.Layer;
import me.yixqiao.jlearn.losses.Loss;
import me.yixqiao.jlearn.matrix.Matrix;

import java.util.ArrayList;

public class Individual {
    /**
     * Layers.
     */
    public final ArrayList<Layer> layers;
    public double score;
    /**
     * Number of layers (including input).
     */
    protected int layerCount;

    public Individual() {
        layers = new ArrayList<>();
    }

    public Individual(ArrayList<Layer> layers) {
        this.layers = layers;
        layerCount = layers.size();
    }

    public Matrix forwardPropagate(Matrix x) {
        Matrix activations = x.clone();
        for (int layerNum = 0; layerNum < layerCount; layerNum++) {
            activations = layers.get(layerNum).forwardPropagate(activations);
        }
        return activations;
    }

    public Individual cloneIndividual() {
        ArrayList<Layer> newLayers = new ArrayList<>();
        for (Layer l : layers)
            newLayers.add(l.cloneLayer());
        return new Individual(newLayers);
    }
}
