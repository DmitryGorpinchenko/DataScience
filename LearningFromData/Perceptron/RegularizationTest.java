import java.io.*;
import java.util.ArrayList;

class RegularizationTest {
	
	static double[][] train; 
	static int[] train_labels;
	static double[][] test; 
	static int[] test_labels;
	
	public static void read_data(String train_file, String test_file) throws IOException {
		ArrayList<String> lines = LinearRegression.read_dataset(train_file);
		int N = lines.size();
		train = new double[N][3];
		train_labels = new int[N];
		String[] tokens = null;
		for(int i = 0; i < N; i++) {
			tokens = lines.get(i).split("\\s+");
			train[i][0] = 1;
			train[i][1] = Double.parseDouble(tokens[0]);
			train[i][2] = Double.parseDouble(tokens[1]);
			train_labels[i] = (int) Double.parseDouble(tokens[2]);
		}
		lines = LinearRegression.read_dataset(test_file);
		N = lines.size();
		test = new double[N][3];
		test_labels = new int[N];
		for(int i = 0; i < N; i++) {
			tokens = lines.get(i).split("\\s+");
			test[i][0] = 1;
			test[i][1] = Double.parseDouble(tokens[0]);
			test[i][2] = Double.parseDouble(tokens[1]);
			test_labels[i] = (int) Double.parseDouble(tokens[2]);
		}
		train = HW6Transformer.transform(train);
		test = HW6Transformer.transform(test);
	}
	
	public static void regularization_test(int k) {
		Model model = LinearRegression.get_model(train, train_labels, Math.pow(10, k));
		double in_error = LinearRegression.calc_error(model, train, train_labels);
		double out_error = LinearRegression.calc_error(model, test, test_labels);
		StdOut.println("In sample error = " + in_error);
		StdOut.println("Out of sample error = " + out_error);
	}
	
	public static void main(String[] args) throws IOException {
		read_data(args[0], args[1]);
		regularization_test(Integer.parseInt(args[2]));
	}
}