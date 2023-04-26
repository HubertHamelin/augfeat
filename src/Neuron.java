import java.util.ArrayList;

enum NeuronType {
    SENSORY,
    HIDDEN,
    ACTION
}

public class Neuron {

    // TODO: no BIAS modelisation in this Neuron construct ?

    ArrayList<Connection> connections;
    String name;
    NeuronType type;
    double output;

    public Neuron(String name, NeuronType type) {
        this.name = name;
        this.type = type;
        this.connections = new ArrayList<Connection>();
    }

    public void addConnection(Connection connection) {
        this.connections.add(connection);
    }

    public void dropConnection(Connection connection) {
        this.connections.remove(connection);
    }

    public void feedForward() {
        // Weighted sum of all connections to the neuron
        double weightedSum = 0;
        for (Connection connection: this.connections) {
            weightedSum += connection.weight * connection.neuron.output;
        }
        // Return the result after processing by the neuron activation function
        this.output = this.activationFunction(weightedSum);
    }

    private double activationFunction(double weightedSum) {
        double activation;
        if (this.type == NeuronType.SENSORY) {
            activation = CustomMath.sigmoid(weightedSum);
        } else {
            activation = Math.tanh(weightedSum);
        }
        return activation;
    }
}
