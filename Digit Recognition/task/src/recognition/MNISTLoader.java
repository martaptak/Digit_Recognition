package recognition;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;
import java.util.Scanner;

class MNISTLoader {

	private static double[][] images;


	static void readImages(File directory) throws IOException {

		images = new double[70000][];

		int index = 0;
		for (File file : Objects.requireNonNull(directory.listFiles())) {
			double[] readFile = readFile(file);
			images[index] = readFile;

			index++;
		}
	}

	private static double[] readFile(File file) throws FileNotFoundException {

		Scanner input = new Scanner(file);

		double[] image = new double[28 * 28 + 1];

		int j = 0;

		for (int row = 0; row < 28; row++) {
			String line = input.nextLine();
			String[] numbers = line.split("\\t");
			int k = 0;
			for (int col = 0; col < 28; col++) {
				image[j] = Double.parseDouble(numbers[k]) / 255d;
				k++;
				j++;
			}
		}

		image[784] = Double.parseDouble(input.nextLine());

		return image;
	}

	static double[][] getImages() {
		return images;
	}




}
