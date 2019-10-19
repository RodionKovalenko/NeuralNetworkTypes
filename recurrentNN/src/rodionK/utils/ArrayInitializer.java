package rodionK.utils;

public class ArrayInitializer {

    // initialize array with 0.0 values
    public static Double [] initializeArray(Double [] array) {
        for (int i = 0; i < array.length; i++) {
            array[i] = 0.0;
        }
        return  array;
    }

    // initialize array with 0.0 values
    public static Double [][] initializeArray(Double [][] array) {
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[0].length; j++) {
                array[i][j] = 0.0;
            }
        }
        return  array;
    }
}
