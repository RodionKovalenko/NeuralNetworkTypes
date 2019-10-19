package rodionK.feedforwardNetwork;

import rodionK.NeuralNetwork;
import rodionK.networkComponents.layer.Layer;
import rodionK.utils.ArrayInitializer;
import rodionK.utils.Propagation;

public class FeedforwardNetwork extends NeuralNetwork {


    public FeedforwardNetwork train2D(Double[][] inputs, Double[][] targets, int numberOfIterations) {
        for (int i = 0; i < numberOfIterations; i++) {
            Propagation.forward2D(this, inputs);
            Propagation.backward2D(this, inputs, targets);
        }

        return this;
    }

    public static void main(String... args) {
        Double[][] inputSet = new Double[4][2];
        Double[][] targets = new Double[4][1];

        inputSet[0] = new Double[]{1.0, 0.0};
        inputSet[1] = new Double[]{1.0, 1.0};
        inputSet[2] = new Double[]{0.0, 0.0};
        inputSet[3] = new Double[]{0.0, 1.0};


        targets[0] = new Double[]{1.0};
        targets[1] = new Double[]{0.0};
        targets[2] = new Double[]{0.0};
        targets[3] = new Double[]{1.0};

        FeedforwardNetwork ffn = new FeedforwardNetwork();

        ffn.setNumberOfInputNeurons(2)
                .setNumberOfHiddenNeurons(10)
                .setNumberOfHiddenLayers(1)
                .setNumberOfOutputNeurons(1)
                .buildDefaultNetwork()
                .setLearningRate(0.6);

        ffn.train2D(inputSet, targets, 5000);
    }

}
