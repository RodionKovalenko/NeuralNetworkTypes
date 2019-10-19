package rodionK.feedforwardNetwork;

import rodionK.NeuralNetwork;
import rodionK.networkComponents.layer.Layer;
import rodionK.utils.Propagation;
import java.util.List;

public class FeedforwardNetwork extends NeuralNetwork {

    public FeedforwardNetwork train2D(Double[][] inputs, Double[][] targets, int numberOfIterations) {

        for (int i = 0; i < numberOfIterations; i++) {
            Propagation.forward2D(this, inputs);
            Propagation.backward2D(this, inputs, targets);
        }

        // print output after training
        List layers = this.getNeuralLayerList();
        Layer layer = (Layer) layers.get(layers.size() - 1);
        for (int i = 0; i < layer.getActivatedOutput().length; i++) {
            for (int j = 0; j < layer.getActivatedOutput()[0].length; j++) {
                System.out.print("Activation of " + layer.getClass() + ": " + layer.getActivatedOutput()[i][j] + ", ");
            }
            System.out.println();
        }
        System.out.println();

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
                .setNumberOfHiddenLayers(2)
                .setNumberOfOutputNeurons(1)
                .buildDefaultNetwork()
                .setLearningRate(0.6);

        ffn.train2D(inputSet, targets, 5000);
    }

}
