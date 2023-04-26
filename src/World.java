import java.awt.*;
import java.util.ArrayList;
import java.lang.Math;
import java.util.Arrays;

enum Direction {
    NORTH,
    SOUTH,
    EAST,
    WEST
}

public class World {

    int xDim;  // Dimension of the world along the x-axis.
    int yDim;  // Dimension of the world along the y-axis.
    int [][] map;  // Agents IDs are located on the world map. Indexes start at 1, empty is 0.
    int nbInitialAgents;  // Number of agents to spawn at each generation.
    int nbConnections;  // Maximal number of active connections in the agents' brain.
    int nbHiddenNeurons;  // Maximal number of active hidden type neurons in the agent's brain.
    int generation;  // Counter of the current generation of the world since its started.
    ArrayList<Agent> agents;  // All agents currently spawned in the world.
    int stepsPerGeneration;
    int limitGeneration;

    public World(int x, int y, int nbInitialAgents, int nbConnections, int nbHiddenNeurons, int limitGeneration,
                 int stepsPerGeneration) {
        this.xDim = x;
        this.yDim = y;
        this.map = new int[y][x];
        this.nbInitialAgents = nbInitialAgents;
        this.nbConnections = nbConnections;
        this.nbHiddenNeurons = nbHiddenNeurons;
        this.agents = new ArrayList<Agent>();
        this.generation = 0;
        this.limitGeneration = limitGeneration;
        this.stepsPerGeneration = stepsPerGeneration;
    }

    public void live() {
        this.populateWorld();
        this.visualize();

        for (int g = 0; g < this.limitGeneration; g++) {
            for (int s = 0; s < this.stepsPerGeneration; s++) {
                for (Agent agent : this.agents) {
                    agent.act(this.map);
                    this.updateMap(agent.id);
                }
                this.visualize();
            }
        }
    }

    private void updateMap(int agentID) {
        int agentIndex = agentID - 1;
        this.map[this.agents.get(agentIndex).previousLoc.y][this.agents.get(agentIndex).previousLoc.x] = 0;
        this.map[this.agents.get(agentIndex).location.y][this.agents.get(agentIndex).location.x] = agentID;
    }

    private void visualize() {
        // Print a rough estimate of the world current state in the console
        System.out.println(" ");
        for (int[] row : this.map) {
            System.out.println(Arrays.toString(row));
        }
    }

    private void populateWorld() {
        for (int i = 0; i < this.nbInitialAgents; i++) {
            // Create a new agent
            Agent agent = this.generateAgent();
            this.agents.add(agent);

            // Update world map
            this.map[agent.location.y][agent.location.x] = agent.id;
        }
    }

    private Point getRandomUnoccupiedLocation() {
        // Is there at least one unoccupied location on the world map ?
        ArrayList<Point> unoccupiedLocations = new ArrayList<Point>();
        for (int j = 0; j < this.map.length; j++) {
            for (int i = 0; i < this.map[j].length; i++) {
                if (map[j][i] == 0) {
                    Point location = new Point(i, j);
                    unoccupiedLocations.add(location);
                }
            }
        }
        // Return a random pick from the subset of unoccupied locations if it exists.
        int randomIndex = (int) (Math.random() * unoccupiedLocations.size());
        return unoccupiedLocations.get(randomIndex);
    }

    private Agent generateAgent() {
        //Spawn an agent at a random unoccupied location on the world map.
        int agentId = this.agents.size() + 1;
        Point location = this.getRandomUnoccupiedLocation();
        return new Agent(agentId, location, this.nbConnections, this.nbHiddenNeurons);
    }
}
