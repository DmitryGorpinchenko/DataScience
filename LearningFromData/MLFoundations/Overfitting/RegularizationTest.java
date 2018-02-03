import java.io.*;
import java.util.ArrayList;

class RegularizationTest {
	
	public static void test1(String train_file, String test_file, double lambda) throws IOException {
		ArrayList<String> lines = Reader.read_data(train_file);
		int N = lines.size();
		int d = lines.get(0).split(" ").length - 1;
		double[][] data = new double[N][d+1];
		int[] labels = new int[N];
		Reader.set_data(lines, data, labels, N, d);
		double[] w = LinearRegression.train(data, labels, lambda);
		double E_in = LinearRegression.calc_error(w, data, labels);
		lines = Reader.read_data(test_file);
		N = lines.size();
		data = new double[N][d+1];
		labels = new int[N];
		Reader.set_data(lines, data, labels, N, d);
		double E_out = LinearRegression.calc_error(w, data, labels);
		StdOut.println("E_in = " + E_in + ", E_out = " + E_out);
	}
	
	public static void test2(String train_file, String test_file) throws IOException {
		ArrayList<String> lines = Reader.read_data(train_file);
		int N = lines.size();
		int d = lines.get(0).split(" ").length - 1;
		double[][] data = new double[N][d+1];
		int[] labels = new int[N];
		Reader.set_data(lines, data, labels, N, d);
		lines = Reader.read_data(test_file);
		N = lines.size();
		double[][] test_data = new double[N][d+1];
		int[] test_labels = new int[N];
		Reader.set_data(lines, test_data, test_labels, N, d);
		for(int k = -10; k <= 2; k++) {
			double lambda = Math.pow(10, k);
			double[] w = LinearRegression.train(data, labels, lambda);
			double E_in = LinearRegression.calc_error(w, data, labels);
			double E_out = LinearRegression.calc_error(w, test_data, test_labels);
			StdOut.println("k = " + k + ": E_in = " + E_in + ", E_out = " + E_out);
		}
	}
	
	public static void main(String[] args) throws IOException {
		//int k = Integer.parseInt(args[2]);
		//double lambda = Math.pow(10, k);
		//test1(args[0], args[1], lambda);
		test2(args[0], args[1]);
	}
}