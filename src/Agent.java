import Sensors.LxSensor;
import Sensors.LySensor;
import Sensors.Sensor;
import java.awt.Point;

enum Action {
    MOVEFORWARD,
    MOVEREVERSE,
    MOVERIGHT,
    MOVELEFT
}


public class Agent {
    // Agent that will populate the environnement
    int id;  // Each agent is identified by a unique ID, starting at 1
    Point location;  // [x, y] coordinates of the agent into the environnement
    Brain brain;  // Artificial brain of the agent, responsible for its actions
    Sensor[] sensors;  // Sensors enable the agent to interpret its environment
    double[] state;  // Vector representing the current state of the agent  //TODO: memory duplicate ?
    Direction direction;  // Current forward movement direction of the agent
    Point previousLoc;  // Location of the agent at the previous step


    public Agent(int id, Point location, int nbConnections, int nbHiddenNeurons) {
        this.id = id;
        this.location = location;
        this.previousLoc = location;
        this.direction = this.initiateRandomDirection();
        this.sensors = this.initiateSensors();
        this.brain = new Brain(nbConnections, nbHiddenNeurons, this.sensors);
    }

    public void updateSensorsState(int[][] environment){
        for (Sensor sensor : this.sensors) {
            sensor.update(environment);
        }
    }

    public void act(int[][] environment){
        // TODO: there must be some code refactorisation here.

        Action action = this.brain.think();
        this.previousLoc = this.location;

        if (action == Action.MOVEFORWARD) {
            Point newLoc = this.moveForward();
            if (this.checkMove(environment, newLoc)) {
                this.location = newLoc;
            }
        } else if (action == Action.MOVELEFT) {
            this.rotateLeft();
            Point newLoc = this.moveForward();
            if (this.checkMove(environment, newLoc)) {
                this.location = newLoc;
            }
        } else if (action == Action.MOVERIGHT) {
            this.rotateRight();
            Point newLoc = this.moveForward();
            if (this.checkMove(environment, newLoc)) {
                this.location = newLoc;
            }
        } else if (action == Action.MOVEREVERSE) {
            this.rotateRight();
            this.rotateRight();
            Point newLoc = this.moveForward();
            if (this.checkMove(environment, newLoc)) {
                this.location = newLoc;
            }
        }
    }

    private Sensor[] initiateSensors() {
        Sensor[] sens = new Sensor[2];
        sens[0] = new LxSensor(this.id);
        sens[1] = new LySensor(this.id);
        return sens;
    }

    private Direction initiateRandomDirection() {
        int randomIndex = (int) (Math.random() * Direction.values().length);
        Direction[] directions = Direction.values();
        return directions[randomIndex];
    }

    // Possible actions by the agent : // TODO: Use a better implementation ?
    private Point moveForward() {
        // TODO: manage blockage between agents and with world elements. if applyAction Do else DoNothing ?
        Point newLoc;
        if (this.direction == Direction.NORTH) {
            newLoc = new Point(this.location.x, this.location.y - 1);
        } else if (this.direction == Direction.SOUTH) {
            newLoc = new Point(this.location.x, this.location.y + 1);
        } else if (this.direction == Direction.EAST) {
            newLoc = new Point(this.location.x + 1, this.location.y);
        } else {
            newLoc = new Point(this.location.x - 1, this.location.y);
        }
        return newLoc;
    }

    private void rotateRight() {
        // Changes the forward direction of the agent after a +90° rotation.
        if (this.direction == Direction.NORTH) {
            this.direction = Direction.EAST;
        } else if (this.direction == Direction.EAST) {
            this.direction = Direction.SOUTH;
        } else if (this.direction == Direction.SOUTH) {
            this.direction = Direction.WEST;
        } else if (this.direction == Direction.WEST) {
            this.direction = Direction.NORTH;
        }
    }

    private void rotateLeft() {
        // Changes the forward direction of the agent after a -90° rotation.
        if (this.direction == Direction.NORTH) {
            this.direction = Direction.WEST;
        } else if (this.direction == Direction.EAST) {
            this.direction = Direction.NORTH;
        } else if (this.direction == Direction.SOUTH) {
            this.direction = Direction.EAST;
        } else if (this.direction == Direction.WEST) {
            this.direction = Direction.SOUTH;
        }
    }

    private boolean checkMove(int[][] environment, Point newLoc) {
        return newLoc.y >= 0 && newLoc.y < environment.length && newLoc.x >= 0 && newLoc.x < environment[0].length
                && environment[newLoc.y][newLoc.x] == 0;
    }
}
