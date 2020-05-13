package testing;

import java.util.ArrayList;
import java.util.Random;

public class Test {
    private static Random random = new Random();
    private static ArrayList<ArrayList<Double>> nWeights = new ArrayList<>();

    public static void main(String[] args) {
        initNetwork(2, 1, 2);
        System.out.println(nWeights);
        System.out.println();
        System.out.println(forward(new ArrayList<Double>() {{
            add(0.0);
            add(1.0);
        }}));
    }

    private static ArrayList<Double> forward(ArrayList<Double> in) {
        ArrayList<Double> inputs = (ArrayList<Double>) in.clone();
        ArrayList<Double> newInputs = null;
        for (int l = 0; l < nWeights.size(); l++) {
            ArrayList<Double> layer = nWeights.get(l);
            newInputs = new ArrayList<Double>();
            for (double neuron : layer) {
                double x = 0;
                for (int i = 0; i < nWeights.get(l).size(); i++) {
                    x += layer.get(i) * nWeights.get(l).get(i);
                }
                x = 1 / (1 + Math.exp(x));
                newInputs.add(x);
            }
            inputs = newInputs;
        }
        return newInputs;
    }

    private static void initNetwork(int in, int hidden, int out) {
        ArrayList<Double> weights1 = new ArrayList<Double>();
        for (int j = 0; j < hidden; j++) {
            for (int i = 0; i < in + 1; i++) {
                weights1.add(random.nextDouble());
            }
        }
        nWeights.add(weights1);

        ArrayList<Double> weights2 = new ArrayList<Double>();
        for (int j = 0; j < out; j++) {
            for (int i = 0; i < hidden + 1; i++) {
                weights2.add(random.nextDouble());
            }
        }
        nWeights.add(weights2);
    }
}
