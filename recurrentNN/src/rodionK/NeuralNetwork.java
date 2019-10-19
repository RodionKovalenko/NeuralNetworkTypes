package rodionK;

import rodionK.networkComponents.layer.HiddenLayer;
import rodionK.networkComponents.layer.Layer;
import rodionK.networkComponents.layer.OutputLayer;
import rodionK.utils.Activation;
import rodionK.utils.WeightInitializer;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    public List<Layer> neuralLayerList;
    public List<List<Double>> networkWeightList2D;


    public Double learningRate;
    public Integer numberOfInputNeurons;
    public Integer numberOfHiddenNeurons;
    public Integer numberOfOutputNeurons;
    public Integer numberOfHiddenLayers;

    public List<List<Double>> getNetworkWeightList2D() {
        return networkWeightList2D;
    }

    public NeuralNetwork setNetworkWeightList2D(List<List<Double>> networkWeightList2D) {
        this.networkWeightList2D = networkWeightList2D;
        return this;
    }

    public Double getLearningRate() {
        return learningRate;
    }

    public NeuralNetwork setLearningRate(Double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public Integer getNumberOfInputNeurons() {
        return numberOfInputNeurons;
    }

    public NeuralNetwork setNumberOfInputNeurons(Integer numberOfInputNeurons) {
        this.numberOfInputNeurons = numberOfInputNeurons;
        return this;
    }

    public Integer getNumberOfHiddenNeurons() {
        return numberOfHiddenNeurons;
    }

    public NeuralNetwork setNumberOfHiddenNeurons(Integer numberOfHiddenNeurons) {
        this.numberOfHiddenNeurons = numberOfHiddenNeurons;
        return this;
    }

    public Integer getNumberOfOutputNeurons() {
        return numberOfOutputNeurons;
    }

    public NeuralNetwork setNumberOfOutputNeurons(Integer numberOfOutputNeurons) {
        this.numberOfOutputNeurons = numberOfOutputNeurons;
        return this;
    }

    public Integer getNumberOfHiddenLayers() {
        return numberOfHiddenLayers;
    }

    public NeuralNetwork setNumberOfHiddenLayers(Integer numberOfHiddenLayers) {
        this.numberOfHiddenLayers = numberOfHiddenLayers;
        return this;
    }

    public List<Layer> getNeuralLayerList() {
        return neuralLayerList;
    }

    public NeuralNetwork setNeuralLayerList(List<Layer> neuralLayerList) {
        this.neuralLayerList = neuralLayerList;
        return this;
    }

    public NeuralNetwork buildDefaultNetwork() {

        if (this.numberOfInputNeurons == null) {
            this.numberOfInputNeurons = 2;
        }
        if (this.numberOfHiddenNeurons == null) {
            this.numberOfHiddenNeurons = 3;
        }
        if (this.numberOfOutputNeurons == null) {
            this.numberOfOutputNeurons = 1;
        }
        if (this.numberOfHiddenLayers == null) {
            numberOfHiddenLayers = 1;
        }


        this.neuralLayerList = new ArrayList<>();
        Integer numberOfInputs = this.numberOfInputNeurons;
        // hidden layers and output layer
        for (int layer = 0; layer < this.numberOfHiddenLayers + 1; layer++) {
            // weights[i][j] from neuron i to neuron j
            // j must be always the number of hidden neurons
            Double[][] layerWeights = new Double[numberOfInputs][numberOfHiddenNeurons];
            Double[] layerBiasArray = new Double[numberOfHiddenNeurons];

            if (layer == this.numberOfHiddenLayers - 1) {
                // number of inputs for output layer
                numberOfInputs = numberOfOutputNeurons;
            } else {
                // number of inputs for  hidden layer
                numberOfInputs = numberOfHiddenNeurons;
            }

            // add layers
            if (layer == this.numberOfHiddenLayers || this.numberOfHiddenLayers == 0) {
                // if output layer
                numberOfInputs = this.numberOfOutputNeurons;
                OutputLayer outputLayer = new OutputLayer();
                layerWeights = new Double[numberOfHiddenNeurons][numberOfInputs];
                outputLayer.setWeights(layerWeights);
                outputLayer.setBias(layerBiasArray);
                Integer previousLayerIndex = layer - 1;

                if (previousLayerIndex >=0) {
                    Layer previousLayer = this.neuralLayerList.get(previousLayerIndex);
                    previousLayer.setNext(outputLayer);
                    outputLayer.setPrevious(previousLayer);
                }

                outputLayer.setActivationType(Activation.activationType.LOGISTIC);
                this.neuralLayerList.add(outputLayer);
            } else {
                // if hidden layer
                HiddenLayer hiddenLayer = new HiddenLayer();
                hiddenLayer.setWeights(layerWeights);
                hiddenLayer.setBias(layerBiasArray);
                if (layer > 0) {
                    Layer previousLayer = this.neuralLayerList.get(layer - 1);
                    previousLayer.setNext(hiddenLayer);
                    hiddenLayer.setPrevious(previousLayer);
                }
                hiddenLayer.setActivationType(Activation.activationType.LOGISTIC);
                this.neuralLayerList.add(hiddenLayer);
            }
        }

        WeightInitializer.initializeWeightMartix2D(this);

        return this;
    }
}
