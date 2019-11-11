package recognition;

import javax.swing.filechooser.FileSystemView;
import java.io.*;
import java.util.Arrays;
import java.util.Scanner;

public class Main {

	private static final String NETWORK_OBJECT_FILENAME = "temp.out";
	private static final String MNIST_FOLDER = FileSystemView.getFileSystemView().getDefaultDirectory().getPath() +
			"\\data\\data";
	private static Scanner scanner = new Scanner(System.in);
	private static Network network;


	public static void main(String[] args) {
		System.out.println(MNIST_FOLDER);
		try {

			MNISTLoader.readImages(new File(MNIST_FOLDER));

		} catch (IOException e) {
			e.printStackTrace();
			return;
		}   
		System.out.println("What do you want?");
		System.out.println("1. Learn the network");
		System.out.println("2. Guess all the numbers");
		System.out.println("3. Guess number from text file");
		System.out.println("Your choice: ");
		int choice = Integer.parseInt(scanner.nextLine());
		pickAChoice(choice);
	}

	private static void pickAChoice(int choice) {

		switch (choice) {
			case 1: { // Teach
				teach();
				break;
			}
			case 2: { // Guess all the numbers
				network = readNetworkFile();
				System.out.println("Guessing...");
				int count = readAllFiles();
				int percent = count / 70000 * 100;
				System.out.println("The network prediction accuracy: " + count + "/70000, " + percent + "%");
				break;
			}
			case 3: { // TEST
				network = readNetworkFile();

				double[]image = readGrid();

				System.out.println("This number is: " + network.calculate(image));
				break;
			}
			default: // Wrong input
				System.out.println("\nWrong input");
		}

	}

	private static void teach() {
		System.out.println("Enter the sizes of the layers:");
		String layers = scanner.nextLine();
		int[] neurons = Arrays.stream(layers.split(" ")).mapToInt(Integer::parseInt).toArray();

		network = new Network(neurons);

		System.out.println("Start training");
		network.train(100, 0.5, 10, 0.15, 0, 0);

		System.out.println("Training done");
		saveNetworkFile();

	}

	private static int readAllFiles() {

		int goods = 0;
		for (int j = 0; j < MNISTLoader.getImages().length; j++) {
			int result = network.calculate(MNISTLoader.getImages()[j]);
			if (result == MNISTLoader.getImages()[j][MNISTLoader.getImages()[0].length - 1]) {
				goods++;
			}
		}

		return goods;
	}


	private static Network readNetworkFile() {
		Network n;

		try {
			FileInputStream fileInputStream = new FileInputStream(NETWORK_OBJECT_FILENAME);

			ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
			n = (Network) objectInputStream.readObject();
			objectInputStream.close();
			fileInputStream.close();

			System.out.println("Network loaded and ready!");

		} catch (Exception e) {
			System.out.println("Error reading network\nNew network");
			e.printStackTrace();
			n = new Network();
		}
		return n;
	}

	private static void saveNetworkFile() {
		try {
			FileOutputStream fileOutputStream = new FileOutputStream(Main.NETWORK_OBJECT_FILENAME);
			ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
			objectOutputStream.writeObject(network);
			objectOutputStream.flush();
			objectOutputStream.close();
			fileOutputStream.close();

			System.out.println("Done! Saved to a file.");
		} catch (Exception e) {
			System.out.println("\nError saving network");
			e.printStackTrace();
		}
	}


	private static double[] readGrid()  {

		double[] image = new double[28 * 28+1];
		System.out.println("Enter filename: ");

		int j = 0;

		for (int row = 0; row < 28; row++) {
			String line = scanner.nextLine();
			String[] numbers = line.split("\\t");
			int k = 0;
			for (int col = 0; col < 28; col++) {
				image[j] = Double.parseDouble(numbers[k]) / 255d;
				k++;
				j++;
			}
		}

		return image;

	}


}
