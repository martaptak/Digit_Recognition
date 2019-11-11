package recognition;

import java.beans.Transient;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

class Network implements Serializable {

	private static final long serialVersionUID = 8011301940005249400L;

	private int numberOfLayers;

	private int[] neuronsInLayers;

	private double[][][] weights;

	private double[][][] deltaWeights;

	private transient double[][] input;

	private transient int inputLength;

	private double[][] neurons;

	private double currentError;

	public Network() {

		this(15, 10);
	}

	public Network(int... neuronsInLayers) {

		this.neuronsInLayers = neuronsInLayers.clone();
		numberOfLayers = neuronsInLayers.length;
		weights = new double[numberOfLayers - 1][][];
		deltaWeights = new double[numberOfLayers - 1][][];
		initWeights();
		neurons = new double[numberOfLayers][];
	}

	private void initWeights() {

		Random random = new Random(serialVersionUID);

		for (int layer = 0; layer < numberOfLayers - 1; layer++) {
			weights[layer] = new double[neuronsInLayers[layer + 1]][];
			deltaWeights[layer] = new double[neuronsInLayers[layer + 1]][];

			for (int i = 0; i < neuronsInLayers[layer + 1]; i++) {
				weights[layer][i] = new double[neuronsInLayers[layer] + 1];
				deltaWeights[layer][i] = new double[neuronsInLayers[layer] + 1];

				for (int j = 0; j < neuronsInLayers[layer] + 1; j++) {
					weights[layer][i][j] = random.nextGaussian();
				}
			}
		}
	}

	public double[][][] copyWeights() {

		double[][][] copy = new double[numberOfLayers - 1][][];
		System.arraycopy(weights, 0, copy, 0, weights.length);
		return copy;
	}

	public void loadMnist() {

		input = MNISTLoader.getImages();
		inputLength = input.length;
	}

	public double[] calculateOutput(double[] input) {

		double[] output = Arrays.copyOf(input, input.length - 1);
		for (int layer = 0; layer < numberOfLayers - 1; layer++) {
			output = Matrix.normalizeNeuron(output, weights[layer]);
		}
		return output;
	}

	public int feedForward(double[] input) {

		neurons[0] = input.clone();
		neurons[0][input.length - 1] = 1.0;

		for (int layer = 0; layer < numberOfLayers - 1; layer++) {
			neurons[layer + 1] = Matrix.normalizeNeuron(neurons[layer], weights[layer]);
			if (layer != numberOfLayers - 2) {
				neurons[layer + 1] = Arrays.copyOf(neurons[layer + 1], neurons[layer + 1].length + 1);
				neurons[layer + 1][neurons[layer + 1].length - 1] = 1.0;
			}
		}

		return (int) input[input.length - 1];
	}

	public void backPropagation(int idealNumber, double eta, double lambda) {

		double[][] error = new double[2][];
		error[0] = new double[neuronsInLayers[numberOfLayers - 1]];

		for (int out = 0; out < neuronsInLayers[numberOfLayers - 1]; out++) {
			error[0][out] = out == idealNumber ? (neurons[numberOfLayers - 1][out] - 1.0) : (neurons[numberOfLayers - 1][out]);
			currentError += error[0][out] * error[0][out] * 0.5;
			error[0][out] *= neurons[numberOfLayers - 1][out] * (1 - neurons[numberOfLayers - 1][out]);

			for (int i = 0; i < weights[numberOfLayers - 2][0].length; i++) {
				deltaWeights[numberOfLayers - 2][out][i] -= eta * (error[0][out] * neurons[numberOfLayers - 2][i] + lambda * weights[numberOfLayers - 2][out][i] / input.length);
			}
		}

		for (int layer = numberOfLayers - 2; layer > 0; layer--) {
			error[1] = error[0].clone();
			error[0] = new double[neuronsInLayers[layer]];
			for (int j = 0; j < neuronsInLayers[layer]; j++) {
				for (int k = 0; k < neuronsInLayers[layer + 1]; k++) {
					error[0][j] += weights[layer][k][j] * error[1][k];
				}
				error[0][j] *= neurons[layer][j] * (1 - neurons[layer][j]);
				for (int i = 0; i < weights[layer - 1][0].length; i++) {
					deltaWeights[layer - 1][j][i] -= eta *
							(error[0][j] * neurons[layer - 1][i] +
									lambda * weights[layer - 1][j][i] / input.length);
				}
			}
		}
	}

	private void recalculateWeights(int batch) {

		for (int layer = 0; layer < (weights.length); layer++) {
			for (int m = 0; m < (weights[layer].length); m++) {
				for (int n = 0; n < (weights[layer][m].length); n++) {
					weights[layer][m][n] += deltaWeights[layer][m][n] / batch;
				}
			}
		}
	}

	public int calculate(double[] values) {

		int digit = -1;
		double bestResult = -1000.0;
		feedForward(values);

		for (int i = 0; i < neuronsInLayers[numberOfLayers - 1]; i++) {
			if (neurons[numberOfLayers - 1][i] > bestResult) {
				bestResult = neurons[numberOfLayers - 1][i];
				digit = i;
			}
		}
		return digit;
	}

	public void train(int epoch, double eta, int batch, double lambda, double minError,
	                  double minDeltaError
	) {
		double previousError;
		double deltaError;
		int epochNumber = 0;
		int trueResult;
		loadMnist();
		List<double[]> numbers = Arrays.asList(input);

		do {
			previousError = currentError;
			double reg = 0;
			currentError = 0;
			trueResult = 0;
			Collections.shuffle(numbers);

			for (int i = 0; i < (inputLength + batch); i += batch) {
				deltaWeights = new double[numberOfLayers - 1][][];

				for (int layer = 0; layer < numberOfLayers - 1; layer++) {
					deltaWeights[layer] = new double[neuronsInLayers[layer + 1]][];
					for (int j = 0; j < neuronsInLayers[layer + 1]; j++) {
						deltaWeights[layer][j] = new double[neuronsInLayers[layer] + 1];
					}
				}
				for (int b = 0; b < batch && i + b < inputLength; b++) {
					int idealNumber = feedForward(numbers.get(i + b));
					backPropagation(idealNumber, eta, lambda);

				}
				recalculateWeights(batch);
			}

			for (double[] number : numbers) {
				if (number[number.length - 1] == calculate(number)) {
					trueResult++;
				}
			}

			currentError /= numbers.size();

			if (Math.abs(lambda) > Float.MIN_VALUE) {
				for (double[][] weight : weights) {
					for (double[] doubles : weight) {
						for (double aDouble : doubles) {
							reg += aDouble * aDouble;
						}
					}
				}
				currentError += lambda * reg / (2 * numbers.size());
			}
			epochNumber++;
			deltaError = Math.abs(currentError - previousError);
			double percent = (double) trueResult * 100 / numbers.size();
			System.out.println("Epoch = " + epochNumber + " currentError = " + currentError + " accuracy = " + percent + "%");
			
		} while (epochNumber < epoch && currentError > minError && deltaError >= minDeltaError);
	}

}
