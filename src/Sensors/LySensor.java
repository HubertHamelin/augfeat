package Sensors;

public class LySensor extends Sensor {

    public LySensor(int agentID){
        this.name = "Ly";
        this.value = 0.0;
        this.agentID = agentID;
    }

    public void update(int[][] environment) {
        // Locate where the agent is on the world map y-axis.
        int yLoc = this.findAgentYaxis(environment);
        // Compute relative location on the y-axis.
        int yAxisLen = environment.length;
        this.value = (double) yLoc / yAxisLen;
    }

    private int findAgentYaxis(int[][] environment) {
        for (int j = 0; j < environment.length; j++) {
            for (int i = 0; i < environment[j].length; i++) {
                if (environment[j][i] == this.agentID) {
                    return j;
                }
            }
        }
        return 0;
    }
}
