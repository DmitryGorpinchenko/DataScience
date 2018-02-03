import java.io.*;
import java.util.ArrayList;

class ValidationTest {
	
	public static void validate(String train, String test) throws IOException {
		ArrayList<String> lines = Reader.read_data(train);
		int N = lines.size();
		int d = lines.get(0).split(" ").length - 1;
		double[][] data = new double[N][d+1];
		int[] labels = new int[N];
		Reader.set_data(lines, data, labels, N, d);
		double[][] train_data = new double[120][d+1];
		int[] train_labels = new int[120];
		Reader.set_data(lines, train_data, train_labels, d, 0, 120);
		double[][] val_data = new double[80][d+1];
		int[] val_labels = new int[80];
		Reader.set_data(lines, val_data, val_labels, d, 120, 200);
		lines = Reader.read_data(test);
		N = lines.size();
		double[][] test_data = new double[N][d+1];
		int[] test_labels = new int[N];
		Reader.set_data(lines, test_data, test_labels, N, d);
		double opt_lambda = 0;
		double min_val_error = Double.POSITIVE_INFINITY; 
		for(int k = 2; k >= -10; k--) {
			double lambda = Math.pow(10, k);
			double[] w = LinearRegression.train(train_data, train_labels, lambda);
			double E_in = LinearRegression.calc_error(w, train_data, train_labels);
			double E_val = LinearRegression.calc_error(w, val_data, val_labels);
			double E_out = LinearRegression.calc_error(w, test_data, test_labels);
			StdOut.println(String.format(java.util.Locale.UK, "k = %3d: E_train = %.3f, E_val = %.3f, E_out = %.3f", k, E_in, E_val, E_out));
			if(min_val_error > E_val) {
				min_val_error = E_val;
				opt_lambda = lambda;
			}
		}
		StdOut.println("\nOptimal lambda = " + opt_lambda + "\n");
		//after validation run on the whole training data set to improve final result
		double[] w = LinearRegression.train(data, labels, opt_lambda);
		double E_in = LinearRegression.calc_error(w, data, labels);
		double E_out = LinearRegression.calc_error(w, test_data, test_labels);
		StdOut.println(String.format(java.util.Locale.UK, "Final hypothesis expected performance: E_in = %.3f, E_out = %.3f", E_in, E_out));
	}
	
	public static void cross_validation(String train, String test, int fold_num) throws IOException {
		ArrayList<String> lines = Reader.read_data(train);
		int data_size = lines.size();
		int fold_size = data_size/fold_num;
		int N = lines.size();
		int d = lines.get(0).split(" ").length - 1;
		double[][] data = new double[N][d+1];
		int[] labels = new int[N];
		Reader.set_data(lines, data, labels, N, d);
		ArrayList<String> test_lines = Reader.read_data(test);
		N = test_lines.size();
		double[][] test_data = new double[N][d+1];
		int[] test_labels = new int[N];
		Reader.set_data(test_lines, test_data, test_labels, N, d);
		double best_E_cv = Double.POSITIVE_INFINITY;
		double opt_lambda = 0;
		for(int k = 2; k >= -10; k--) {
			double lambda = Math.pow(10, k);
			double E_cv = 0;
			for(int f = 0; f < fold_num; f++) {
				ArrayList<String> train_lines = new ArrayList<>();
				ArrayList<String> val_lines = new ArrayList<>();
				int lo = fold_size*f;
				int hi = fold_size*(f+1);
				for(int i = 0; i < data_size; i++) {
					if(i >= lo && i < hi) {
						val_lines.add(lines.get(i));
					} else {
						train_lines.add(lines.get(i));
					}	
				}
				double[][] train_data = new double[data_size - fold_size][d+1];
				int[] train_labels = new int[data_size - fold_size];
				Reader.set_data(train_lines, train_data, train_labels, data_size - fold_size, d);
				double[][] val_data = new double[fold_size][d+1];
				int[] val_labels = new int[fold_size];
				Reader.set_data(val_lines, val_data, val_labels, fold_size, d);
				double[] w = LinearRegression.train(train_data, train_labels, lambda);
				E_cv += LinearRegression.calc_error(w, val_data, val_labels);
			}
			E_cv /= fold_num;
			if(E_cv < best_E_cv) {
				best_E_cv = E_cv;
				opt_lambda = lambda;
			}
		}
		StdOut.println("\nOptimal lambda = " + opt_lambda + ", E_cv = " + best_E_cv + "\n");
		//after validation run on the whole training data set to improve final result
		double[] w = LinearRegression.train(data, labels, opt_lambda);
		double E_in = LinearRegression.calc_error(w, data, labels);
		double E_out = LinearRegression.calc_error(w, test_data, test_labels);
		StdOut.println(String.format(java.util.Locale.UK, "Final hypothesis expected performance: E_in = %.3f, E_out = %.3f", E_in, E_out));
	}
	
	public static void main(String[] args) throws IOException {
		//validate(args[0], args[1]);
		cross_validation(args[0], args[1], Integer.parseInt(args[2])); 
	}
}