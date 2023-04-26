import Sensors.Sensor;

import java.util.Objects;


public class Brain {

    // TODO : handling computations using Matrices will be faster.
    // TODO: nbConnections inherited from the agent's genome. At first we will use a simple default brain.
    // TODO: current implementation might work for small brains, but more efficiency is needed to expand its size.

    Neuron[] sensoryNeurons;
    Neuron[] hiddenNeurons;
    Neuron[] actionNeurons;
    int nbConnections;
    Neuron[][] layers;

    public Brain(int nbConnections, int nbHiddenNeurons, Sensor[] sensors) {
        this.generateSensoryNeurons(sensors);
        this.generateHiddenNeurons(nbHiddenNeurons);
        this.generateActionNeurons();
        this.nbConnections = nbConnections;
        this.generateRandomConnections();
        this.exportBrain();
    }

    public Action think() {
        // test : the brain only know how to move forward.
        // TODO: interprets actionNeurons outputs as probabilities and select the resulting action for the agent.
        return Action.MOVEFORWARD;
    }

    public void exportBrain() {
        for (Neuron neuron : this.hiddenNeurons) {
            for (Connection connection : neuron.connections) {
                System.out.println(connection.neuron.name + " " + neuron.name + " " + connection.weight);
            }
        }
        for (Neuron neuron : this.actionNeurons) {
            for (Connection connection : neuron.connections) {
                System.out.println(connection.neuron.name + " " + neuron.name + " " + connection.weight);
            }
        }
    }

    private void generateSensoryNeurons(Sensor[] sensors) {
        this.sensoryNeurons = new Neuron[sensors.length];
        for (int i = 0; i < sensors.length; i++) {
            this.sensoryNeurons[i] = new Neuron(sensors[i].name, NeuronType.SENSORY);
        }
    }

    private void generateHiddenNeurons(int nbHiddenNeurons) {
        this.hiddenNeurons = new Neuron[nbHiddenNeurons];
        for (int i = 0; i < nbHiddenNeurons; i++) {
            this.hiddenNeurons[i] = new Neuron("N" + i, NeuronType.HIDDEN);
        }
    }

    private void generateActionNeurons() {
        this.actionNeurons = new Neuron[Action.values().length];
        for (int i = 0; i < Action.values().length; i++) {
            this.actionNeurons[i] = new Neuron(String.valueOf(Action.values()[i]), NeuronType.ACTION);
        }
    }

    private void generateRandomConnections() {
        for (int i = 0; i < this.nbConnections; i++) {
            Connection connection;

            // Generate a random weight for initialisation : TODO: this will be done by the Genome translation.
            double randomWeight = Math.random() * 4;
            if (Math.random() >= 0.5){
                randomWeight *= -1;
            }

            // Pick at random a sensoryNeuron or a hiddenNeuron that will be the source of the connection
            int randomStartIndex = (int) (Math.random() * (this.sensoryNeurons.length + this.hiddenNeurons.length));
            if (randomStartIndex >= this.sensoryNeurons.length) {
                randomStartIndex -= this.sensoryNeurons.length;
                connection = new Connection(randomWeight, this.hiddenNeurons[randomStartIndex]);
            } else {
                connection = new Connection(randomWeight, this.sensoryNeurons[randomStartIndex]);
            }

            // Pick at random either a hiddenNeuron or an actionNeuron that will be at the end of the connection.
            int randomEndIndex = (int) (Math.random() * (this.hiddenNeurons.length + this.actionNeurons.length));
            if (randomEndIndex >= this.hiddenNeurons.length) {
                randomEndIndex -= this.hiddenNeurons.length;
                this.actionNeurons[randomEndIndex].addConnection(connection);
            } else {
                this.hiddenNeurons[randomEndIndex].addConnection(connection);
            }
        }
    }

    private void simplifyNetwork() {
        // Get rid of connections between hidden and action neuron if hidden has no path to sensory
        for (Neuron neuron : this.actionNeurons) {
            for (Connection connection : neuron.connections) {
                if (!this.hasPathToSensory(connection, neuron.name)) {
                    neuron.dropConnection(connection);
                }
            }
        }

        // Get rid of connections between hidden and sensory if hidden has no path to action
        // Get rid of connections between hidden and hidden if no path to action
        // Get rid of neurons that are not involved in any connection
    }

    private void organizeLayers() {
        // Use established connections to sort each neuron to its corresponding layer. Mandatory for feed-forward op.
    }

    private void feedForward() {

    }

    private boolean hasPathToSensory(Connection connection, String neuronName) {
        // TODO: might be some code refactorisation with the search launch (first round included in recursion ?)
        if (connection.neuron.type == NeuronType.SENSORY) {
            return true;
        } else {
            // if no more connections : return false
            if (connection.neuron.connections.size() == 0) {
                return false;
            }
            // else : pursue recursive search.
            // exclude a self-connection from the search (infinite loop)
            if (!Objects.equals(neuronName, connection.neuron.name)) {
                for (Connection next : connection.neuron.connections) {
                    if (this.hasPathToSensory(next, connection.neuron.name)) {
                        return true;
                    }
                }
            }
            return false;
        }
    }

    // private boolean hasPathToAction() {}
}
