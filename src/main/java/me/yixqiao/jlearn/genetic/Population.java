package me.yixqiao.jlearn.genetic;

import me.yixqiao.jlearn.exceptions.NeuralNetworkException;
import me.yixqiao.jlearn.layers.InputLayer;
import me.yixqiao.jlearn.layers.Layer;
import me.yixqiao.jlearn.losses.Loss;
import me.yixqiao.jlearn.models.Model;

import java.util.ArrayList;

public class Population {
    protected final int indivCount;
    protected ArrayList<Individual> individuals;

    protected final ArrayList<Layer> layers;
    protected int layerCount;

    public Population(int indivCount) {
        this.indivCount = indivCount;
        individuals = new ArrayList<>();
        layers = new ArrayList<>();
    }


    public Population addLayer(Layer layer) {
        layers.add(layer);
        return this;
    }

    public void prepLayers() {
        layerCount = layers.size();
        if (!(layers.get(0) instanceof InputLayer)) {
            throw new NeuralNetworkException("First layer is not an input layer");
        }
        for (int layer = 0; layer < layerCount - 1; layer++) {
            layers.get(layer + 1).initLayer(layers.get(layer).getOutSize(), layers.get(layer).getActivation());
        }
    }

    public void initPop() {
        individuals = new ArrayList<>();
        for(int i=0; i<indivCount; i++){
            individuals.add(new Individual(layers)); // Clone layers
        }
    }
}
