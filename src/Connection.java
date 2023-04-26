public class Connection {

    Neuron neuron;
    double weight;

    public Connection(double weight, Neuron neuron) {
        this.weight = weight;
        this.neuron = neuron;
    }

    public void updateWeight() {}
}
