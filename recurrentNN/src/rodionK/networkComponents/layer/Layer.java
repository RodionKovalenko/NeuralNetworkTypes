package rodionK.networkComponents.layer;

import rodionK.utils.Activation;

public class Layer {
    Double [][] weights;
    Double [] bias;
    Double [][] inactivatedOutput;
    Double[][] activatedOutput;
    Double[][] errors;
    Double[][] gradient;
    Double heavisideThreshthold;
    Layer next;
    Layer previous;
    public Activation.activationType activationType;

    public Double[][] getWeights() {
        return weights;
    }

    public Layer setWeights(Double[][] weights) {
        this.weights = weights;
        return this;
    }

    public Layer getNext() {
        return next;
    }

    public Layer setNext(Layer next) {
        this.next = next;
        return this;
    }

    public Layer getPrevious() {
        return previous;
    }

    public Layer setPrevious(Layer previous) {
        this.previous = previous;
        return this;
    }

    public Double[] getBias() {
        return bias;
    }

    public Layer setBias(Double[] bias) {
        this.bias = bias;
        return this;
    }

    public Double[][] getInactivatedOutput() {
        return inactivatedOutput;
    }

    public Layer setInactivatedOutput(Double[][] inactivatedOutput) {
        this.inactivatedOutput = inactivatedOutput;
        return this;
    }

    public Double[][] getActivatedOutput() {
        return activatedOutput;
    }

    public Layer setActivatedOutput(Double[][] activatedOutput) {
        this.activatedOutput = activatedOutput;
        return this;
    }

    public Double[][] getErrors() {
        return errors;
    }

    public Layer setErrors(Double[][] errors) {
        this.errors = errors;
        return this;
    }

    public Double[][] getGradient() {
        return gradient;
    }

    public Layer setGradient(Double[][] gradient) {
        this.gradient = gradient;
        return this;
    }

    public Activation.activationType getActivationType() {
        return activationType;
    }

    public Layer setActivationType(Activation.activationType activationType) {
        this.activationType = activationType;
        return this;
    }

    public Double getHeavisideThreshthold() {
        return heavisideThreshthold;
    }

    public Layer setHeavisideThreshthold(Double heavisideThreshthold) {
        this.heavisideThreshthold = heavisideThreshthold;
        return this;
    }
}
