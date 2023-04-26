public class CustomMath {

    public static double sigmoid(double t) {
        return 1 / (1 + Math.pow(Math.E, (-1 * t)));
    }

}
