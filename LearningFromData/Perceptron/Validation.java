import java.io.*;
import java.util.ArrayList;

class Validation {
	
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
	
	public static void validation(int begin_id, int end_id) {
		int N = end_id - begin_id + 1;
		double[][] train_data = new double[N][];
		int[] train_data_labels = new int[N];
		for(int i = begin_id; i <= end_id; i++) {
			train_data[i-begin_id] = train[i];
			train_data_labels[i - begin_id] = train_labels[i];
		}	
		int val = train.length-N;
		double[][] validation_data = new double[val][];
		int[] validation_data_labels = new int[val];
		for(int i = 0; i < begin_id; i++) {
			validation_data[i] = train[i];
			validation_data_labels[i] = train_labels[i];
		}
		for(int i = end_id+1; i < train.length; i++) {
			validation_data[i-end_id-1+begin_id] = train[i];
			validation_data_labels[i-end_id-1+begin_id] = train_labels[i];
		}
		for(int k = 3; k <= 7; k++) {
			double[][] transformed_train_data = HW7Transformer.transform(train_data, k);
			double[][] transformed_valid_data = HW7Transformer.transform(validation_data, k);
			Model model = LinearRegression.get_model(transformed_train_data, train_data_labels);
			double validation_error = LinearRegression.calc_error(model, transformed_valid_data, validation_data_labels);
			double out_error = LinearRegression.calc_error(model, test, test_labels);
			StdOut.println("\nValidation error for    k = " + k + ": " + validation_error);
			StdOut.println("Out of sample error for k = " + k + ": " + out_error);
		}
	}
	
	public static void main(String[] args) throws IOException {
		read_data(args[0], args[1]);
		validation(Integer.parseInt(args[2]), Integer.parseInt(args[3]));
	}
}