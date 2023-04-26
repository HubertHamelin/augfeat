package Sensors;

public abstract class Sensor {
    /*
        Lx : east/west world location
        Ly : north/south world location
     */
    public String name;
    public double value;
    protected int agentID;

    public abstract void update(int[][] environment);
}