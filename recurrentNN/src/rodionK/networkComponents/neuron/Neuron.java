package rodionK.networkComponents.neuron;

import java.util.List;

public abstract class Neuron {
    public Double weightArray[];
    public List<Double> weightList;
    public Neuron previous;
    public Neuron next;

    public Double[] getWeightArray() throws Exception {
        if (weightArray != null) {
            return weightArray;
        }

        if (weightList != null) {
            Double[] weights = new Double[weightList.size()];


            for (int i = 0; i < weights.length; i++) {
                weights[i] = this.weightList.get(i);
            }

            return weights;
        }

        throw new Exception("Weight array is null");
    }


    public Neuron setWeightArray(Double[] weightArray) {
        this.weightArray = weightArray;
        return this;
    }

    public List<Double> getWeightList() {
        return weightList;
    }

    public Neuron setWeightList(List<Double> weightList) {
        this.weightList = weightList;
        return this;
    }

    public Neuron getPrevious() {
        return previous;
    }

    public Neuron setPrevious(Neuron previous) {
        this.previous = previous;
        return this;
    }

    public Neuron getNext() {
        return next;
    }

    public Neuron setNext(Neuron next) {
        this.next = next;
        return this;
    }
}
