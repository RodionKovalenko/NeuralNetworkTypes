package rodionK.utils;

import rodionK.NeuralNetwork;
import rodionK.networkComponents.layer.Layer;

import java.util.List;

public class Activation {
    public enum activationType {
        IDENTITY,
        LOGISTIC,
        TANH,
        HEAVISIDE,
        RELU,
        MEAN_LINEAR,
        ENFORCING
    }

    public static Double[] activateOutput(Layer layer, Double[] inactivatedOutput) {

        switch (layer.getActivationType()) {
            case IDENTITY:

                break;
            case LOGISTIC:
                return Activation.logisticActivation(inactivatedOutput);
            case TANH:

                break;
            case HEAVISIDE:

                break;
            case RELU:

                break;
            case ENFORCING:
            default:
                return null;

        }

        return null;
    }

    public static Double[][] activateOutput(Layer layer, Double[][] inactivatedOutput) {

        switch (layer.getActivationType()) {
            case IDENTITY:

                break;
            case LOGISTIC:
                return Activation.logisticActivation(inactivatedOutput);
            case TANH:

                break;
            case HEAVISIDE:

                break;
            case RELU:

                break;
            case ENFORCING:
            default:
                return null;

        }

        return null;
    }

    public static Double[][] logisticActivation(Double[][] inactivatedOutput) {
        Double[][] activatedOutput = new Double[inactivatedOutput.length][inactivatedOutput[0].length];

        for (int i = 0; i < activatedOutput.length; i++) {
            for (int j = 0; j < activatedOutput[0].length; j++) {
                activatedOutput[i][j] = 1.0 / (1 + Math.exp(-inactivatedOutput[i][j]));
            }
        }

        return activatedOutput;
    }

    public static Double[] logisticActivation(Double[] inactivatedOutput) {
        Double[] activatedOutput = new Double[inactivatedOutput.length];

        for (int i = 0; i < activatedOutput.length; i++) {
            activatedOutput[i] = 1.0 / (1 + Math.exp(-inactivatedOutput[i]));
        }

        return activatedOutput;
    }


    public static Double[][] tanhActivation(Double[][] inactivatedOutput) {
        Double[][] activatedOutput = new Double[inactivatedOutput.length][inactivatedOutput[0].length];

        for (int i = 0; i < activatedOutput.length; i++) {
            for (int j = 0; j < activatedOutput[0].length; j++) {
                activatedOutput[i][j] = (1.0 - Math.exp(-2 * inactivatedOutput[i][j])) / (1.0 + Math.exp(-2 * inactivatedOutput[i][j]));
            }
        }

        return activatedOutput;
    }

    public static Double[][] heavisideActivation(Double[][] inactivatedOutput, Double threshold) {
        Double[][] activatedOutput = new Double[inactivatedOutput.length][inactivatedOutput[0].length];

        if (threshold == null) {
            threshold = 0.5;
        }

        for (int i = 0; i < activatedOutput.length; i++) {
            for (int j = 0; j < activatedOutput[0].length; j++) {
                if (inactivatedOutput[i][j] >= threshold) {
                    activatedOutput[i][j] = inactivatedOutput[j][j];
                } else {
                    activatedOutput[i][j] = 0.0;
                }
            }
        }

        return activatedOutput;
    }


    public static Double[][] reluActivation(Double[][] inactivatedOutput) {
        Double[][] activatedOutput = new Double[inactivatedOutput.length][inactivatedOutput[0].length];

        for (int i = 0; i < activatedOutput.length; i++) {
            for (int j = 0; j < activatedOutput[0].length; j++) {
                if (inactivatedOutput[i][j] > 0) {
                    activatedOutput[i][j] = inactivatedOutput[j][j];
                } else {
                    activatedOutput[i][j] = 0.0;
                }
            }
        }

        return activatedOutput;
    }

    public static Double[][] enforcingActivation(Double[][] weights, Double[][] inactivatedOutput) {
        Double[][] activatedOutput = new Double[inactivatedOutput.length][inactivatedOutput[0].length];
        Double k = weights.length * 1.0;


        for (int j = 0; j < activatedOutput[0].length; j++) {
            Double sum = 0.0;
            for (int i = 0; i < activatedOutput.length; i++) {
                sum += (weights[i][j] * inactivatedOutput[i][j]) / 1 + (Math.abs(weights[i][j] * inactivatedOutput[i][j]));
            }
            activatedOutput[j][0] = sum;
        }

        return activatedOutput;
    }
}
