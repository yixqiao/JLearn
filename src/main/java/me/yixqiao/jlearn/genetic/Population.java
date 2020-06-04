package me.yixqiao.jlearn.genetic;

import me.yixqiao.jlearn.exceptions.NeuralNetworkException;
import me.yixqiao.jlearn.layers.InputLayer;
import me.yixqiao.jlearn.layers.Layer;
import me.yixqiao.jlearn.losses.Loss;
import me.yixqiao.jlearn.matrix.Matrix;
import me.yixqiao.jlearn.models.Model;

import java.util.ArrayList;

public class Population {
    protected final int indivCount;
    protected final ArrayList<Layer> layers;
    protected Individual[] individuals;
    protected int layerCount;

    public Population(int indivCount) {
        this.indivCount = indivCount;
        layers = new ArrayList<>();
        individuals = new Individual[indivCount];
    }


    public Population addLayer(Layer layer) {
        layers.add(layer);
        return this;
    }

    public void initLayers() {
        layerCount = layers.size();
        if (!(layers.get(0) instanceof InputLayer)) {
            throw new NeuralNetworkException("First layer is not an input layer");
        }
        for (int layer = 0; layer < layerCount - 1; layer++) {
            layers.get(layer + 1).initLayer(layers.get(layer).getOutSize(), layers.get(layer).getActivation());
        }
    }

    public void initPop() {
        for (int i = 0; i < indivCount; i++) {
            ArrayList<Layer> iLayers = new ArrayList<>();
            for (Layer l : layers)
                iLayers.add(l.cloneSettings());
            individuals[i] = new Individual(iLayers);
        }
    }

    public Matrix predict(int indivNum, Matrix x) {
        return individuals[indivNum].forwardPropagate(x);
    }


}
