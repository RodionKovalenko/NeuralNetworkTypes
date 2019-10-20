package rodionK.utils;

import rodionK.NeuralNetwork;
import rodionK.networkComponents.layer.Layer;

import javax.xml.crypto.dom.DOMCryptoContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class WeightInitializer {

    /* initialize weights matrices with random values in each hidden and output layer */
    public static NeuralNetwork initializeWeightMartix2D(NeuralNetwork network) {
        List<Layer> layers = network.getNeuralLayerList();

        if (layers != null) {

            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                Layer layer = layers.get(layerIndex);
                Double[][] weightMatrix2D = layer.getWeights();
                Double[] biasArray = layer.getBias();
                List<List<Double>> listWeights = layer.getListWeights();

                if (listWeights == null) {
                    listWeights = new ArrayList<>();
                }

                if (weightMatrix2D.length != 0) {
                    for (int j = 0; j < weightMatrix2D[0].length; j++) {
                        ArrayList<Double> layerWeightList = new ArrayList<>();
                        biasArray[j] = new Random().nextGaussian() * 0.3;

                        for (int i = 0; i < weightMatrix2D.length; i++) {
                            weightMatrix2D[i][j] = new Random().nextGaussian() * 0.3;

                            layerWeightList.add(weightMatrix2D[i][j]);
                        }

                        listWeights.add(layerWeightList);
                    }

                    layer.setListWeights(listWeights);
                }
            }
        }

        network.setNeuralLayerList(layers);
        return network;
    }
}
