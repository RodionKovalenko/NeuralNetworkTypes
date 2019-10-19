package rodionK.utils;

import rodionK.NeuralNetwork;
import rodionK.networkComponents.layer.Layer;

import java.util.List;

public class Derivation {

    public static Double getDerivative(NeuralNetwork network, Double activatedOutput) {
        List<Layer> layers = network.getNeuralLayerList();

        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            Layer layer = layers.get(layerIndex);
            switch (layer.getActivationType()) {
                case IDENTITY:

                    break;
                case LOGISTIC:
                    return activatedOutput * (1 - activatedOutput);
                case TANH:

                    break;
                case HEAVISIDE:

                    break;
                case RELU:

                    break;
                case ENFORCING:

            }

        }

        return null;
    }
}
