package recognition;

public class Matrix {

	public static double[] multiply(double[] vector, double[][] matrix) {
		if (vector.length != matrix[0].length) {
			throw new IllegalArgumentException("Illegal vector length");
		}

		double[] newVector = new double[matrix.length];
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[0].length; j++) {
				newVector[i] += vector[j] * matrix[i][j];
			}
		}

		return newVector;
	}

	public static double[] normalizeNeuron(double[] vector, double[][] matrix) {
		if (vector.length != matrix[0].length) {
			throw new IllegalArgumentException("Illegal vector length");
		}

		double[] newVector = new double[matrix.length];
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[0].length; j++) {
				newVector[i] += vector[j] * matrix[i][j];
			}

			newVector[i] = sigmoidFunction(newVector[i]);
		}

		return newVector;
	}

	private static double sigmoidFunction(double v) {
		return 1 / (1 + Math.pow(Math.E, -v));
	}

	private static double[][] sigmoidFunction(double[][] matrix) {
		double[][] normalizeMatrix = new double[matrix.length][];
		for (int i = 0; i < matrix.length; i++) {
			normalizeMatrix[i] = matrix[i].clone();
			for (int j = 0; j < matrix[i].length; j++) {
				normalizeMatrix[i][j] = sigmoidFunction(normalizeMatrix[i][j]);
			}
		}
		return normalizeMatrix;
	}

	public static double[][][] sigmoidFunction(double[][][] matrix) {
		double[][][] normalizeMatrix = new double[matrix.length][][];
		for (int i = 0; i < matrix.length; i++) {
			normalizeMatrix[i] = matrix[i].clone();
			for (int j = 0; j < matrix[i].length; j++) {
				normalizeMatrix[i][j] = matrix[i][j].clone();
				for (int k = 0; k < matrix[i][j].length; k++) {
					normalizeMatrix[i][j][k] = sigmoidFunction(normalizeMatrix[i][j][k]);
				}
			}
		}
		return normalizeMatrix;
	}

	public static double derivativeOfSigmoid (double x) {

		return sigmoidFunction(x)*(1-sigmoidFunction(x));
	}

	public static double [][] derivativeOfSigmoid (double [][] matrix) {

		double[][] result = new double[matrix.length][];
		for(int i = 0;i<matrix.length;i++) {
			result[i] = matrix[i].clone();
			for(int j = 0;j<matrix[i].length;j++) {
				result[i][j]=derivativeOfSigmoid(result[i][j]);
			}
		}
		return result;
	}

	public static double [][][] derivativeOfSigmoid (double [][][] matrix) {

		double[][][] result = new double[matrix.length][][];
		for(int i = 0;i<matrix.length;i++) {
			result[i] = matrix[i].clone();
			for(int j = 0;j<matrix[i].length;j++) {
				result[i][j] = matrix[i][j].clone();
				for(int k = 0;k<matrix[i][j].length;k++) {
					result[i][j][k]=derivativeOfSigmoid(result[i][j][k]);
				}
			}
		}
		return result;
	}

	public static double[] divide(double[] vector, double[][] matrix) {
		if (vector.length != matrix[0].length) {
			throw new IllegalArgumentException("Illegal vector length");
		}

		double[] newVector = new double[matrix.length];
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[0].length; j++) {
				newVector[i] += vector[j] / matrix[i][j];
			}
		}

		return newVector;
	}

	public static double [][] transpose (double [][] matrix) {
		if(matrix[0].length!=matrix[1].length) {
			throw new IllegalArgumentException("The matrix is not rectangle!");
		}
		double[][] transposedMatrix = new double[matrix[0].length][matrix.length];
		for(int i = 0;i<matrix.length;i++) {
			for(int j = 0;j<matrix[0].length;j++) {
				transposedMatrix[j][i]=matrix[i][j];
			}
		}
		return transposedMatrix;
	}

}
