package me.yixqiao.jlearn.genetic;

import me.yixqiao.jlearn.layers.Layer;

import java.util.ArrayList;

public class Individual {
    /**
     * Layers.
     */
    protected final ArrayList<Layer> layers;

    /**
     * Number of layers (including input).
     */
    protected int layerCount;

    public Individual() {
        layers = new ArrayList<>();
    }

    public Individual(ArrayList<Layer> layers) {
        this.layers = layers;
    }
}
