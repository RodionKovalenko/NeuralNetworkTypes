package rodionK.utils;

import rodionK.NeuralNetwork;
import rodionK.feedforwardNetwork.FeedforwardNetwork;
import rodionK.networkComponents.layer.Layer;

import java.util.List;

public class Propagation {

    public static NeuralNetwork forward2D(NeuralNetwork network, Double[][] inputWeigths) {
        if (network instanceof FeedforwardNetwork) {
            return forward2DFeedforward(network, inputWeigths);
        }

        return network;
    }

    public static NeuralNetwork backward2D(NeuralNetwork network, Double[][] inputs, Double[][] targets) {
        if (network instanceof FeedforwardNetwork) {
            return backward2DFeedForward(network, inputs, targets);
        }

        return network;
    }


    public static NeuralNetwork forward2DFeedforward(NeuralNetwork network, Double[][] inputSet) {
        List<Layer> layers = network.getNeuralLayerList();
        Double[][] output = inputSet.clone();

        if (layers != null) {
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                Layer layer = layers.get(layerIndex);
                output = getLayerOutput(layer, output);
            }
        }

        return network;
    }

    public static Double[][] getLayerOutput(Layer layer, Double[][] inputSet) {
        Double[][] output = null;

        for (int inputSetNumber = 0; inputSetNumber < inputSet.length; inputSetNumber++) {
            Double[] input = inputSet[inputSetNumber];
            Double[][] weightMatrix2D = layer.getWeights();
            Double[] bias = layer.getBias();
            output = layer.getActivatedOutput();

            if (output == null) {
                output = new Double[inputSet.length][weightMatrix2D[0].length];
                output = ArrayInitializer.initializeArray(output);
            }

            for (int j = 0; j < weightMatrix2D[0].length; j++) {
                for (int i = 0; i < weightMatrix2D.length; i++) {
                    output[inputSetNumber][j] += weightMatrix2D[i][j] * input[i];
                }

                output[inputSetNumber][j] += bias[j];
            }

            output = Activation.activateOutput(layer, output);
        }

        layer.setActivatedOutput(output);

        return output;
    }

    public static NeuralNetwork backward2DFeedForward(NeuralNetwork network, Double[][] inputSet, Double[][] targetSet) {
        List<Layer> layers = network.getNeuralLayerList();

        for (int layerIndex = layers.size() - 1; layerIndex >= 0; layerIndex--) {
            Layer layer = layers.get(layerIndex);
            Double[][] activatedOutput = layer.getActivatedOutput();
            Double[][] previousLayerInput = null;
            Integer nextIndex = layerIndex == layers.size() - 1 ? layerIndex : layerIndex + 1;
            Double[][] gradientY = layer.getGradient();
            Layer nextLayer = layers.get(nextIndex);
            Double[][] layerWeights = layer.getWeights();
            Double[] bias = layer.getBias();
            Double[][] layerWeightsNextLayer = nextLayer.getWeights();
            Double[][] errors = layer.getErrors();
            Double[][] errorsNextLayer = nextLayer.getErrors();
            Layer previousLayer = null;

            if (gradientY == null) {
                gradientY = ArrayInitializer.initializeArray(new Double[layerWeights.length][layerWeights[0].length]);
            }
            if (errors == null) {
                errors = ArrayInitializer.initializeArray(new Double[targetSet.length][layerWeights[0].length]);
            }

            if (layerIndex > 0) {
                previousLayer = layers.get(layerIndex - 1);
                previousLayerInput = previousLayer.getActivatedOutput();
            } else {
                previousLayerInput = inputSet;
            }

            if (layerIndex == layers.size() - 1) {
                // calculate gradients of output layer
                // calculate errors for all data sets
                for (int i = 0; i < targetSet.length; i++) {
                    for (int j = 0; j < targetSet[0].length; j++) {
                        errors[i][j] = targetSet[i][j] - activatedOutput[i][j];
                    }
                }

                for (int inputSetNumber = 0; inputSetNumber < inputSet.length; inputSetNumber++) {
                    for (int j = 0; j < layerWeights[0].length; j++) {
                        for (int i = 0; i < layerWeights.length; i++) {
                            gradientY[i][j] = (errors[inputSetNumber][j]
                                    * previousLayerInput[inputSetNumber][j]);

                            layerWeights[i][j] += (network.getLearningRate() * gradientY[i][j]) / inputSet.length;
                        }

                        bias[j] += (network.getLearningRate() * errors[inputSetNumber][j]) / inputSet.length;
                    }
                }

                // update errors in layer
                layer.setErrors(errors);
            } else {
                // calculate gradients of hidden layers
                for (int inputSetNumber = 0; inputSetNumber < inputSet.length; inputSetNumber++) {

                    for (int j = 0; j < layerWeights[0].length; j++) {

                        for (int k = 0; k < errorsNextLayer[0].length; k++) {
                            errors[inputSetNumber][j] += errorsNextLayer[inputSetNumber][k] * layerWeightsNextLayer[j][k];
                        }

                        errors[inputSetNumber][j] *= Derivation.getDerivative(network, activatedOutput[inputSetNumber][j]);

                        for (int i = 0; i < layerWeights.length; i++) {
                            gradientY[i][j] = (previousLayerInput[inputSetNumber][i]
                                    * errors[inputSetNumber][j]);

                            layerWeights[i][j] += (network.getLearningRate() * gradientY[i][j]) / inputSet.length;
                        }

                        bias[j] += (network.getLearningRate()
                                * errors[inputSetNumber][j]) / inputSet.length;
                    }
                }
                // update error for current layer
                layer.setErrors(errors);
            }
        }

        // remove Errors
        for (int i = 0; i < layers.size(); i++) {
            layers.get(i).setErrors(null);
            layers.get(i).setGradient(null);
        }

        return network;
    }
}
