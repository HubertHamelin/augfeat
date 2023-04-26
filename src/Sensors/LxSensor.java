package Sensors;

public class LxSensor extends Sensor {

    public LxSensor(int agentID){
        this.name = "Lx";
        this.value = 0.0;
        this.agentID = agentID;
    }

    public void update(int[][] environment) {
        // Locate where the agent is on the world map x-axis.
        int xLoc = this.findAgentXaxis(environment);
        // Compute relative location on the x-axis.
        int xAxisLen = environment[0].length;
        this.value = (double) xLoc / xAxisLen;
    }

    private int findAgentXaxis(int[][] environment) {
        for (int[] ints : environment) {
            for (int i = 0; i < ints.length; i++) {
                if (ints[i] == this.agentID) {
                    return i;
                }
            }
        }
        return 0;
    }
}
