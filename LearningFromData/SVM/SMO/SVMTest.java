import java.io.*;
import java.util.ArrayList;

class SVMTest {
	
	public static void test_poly(String train, String test) throws IOException {
		ArrayList<String> train_lines = Reader.read_data(train);
		int N = train_lines.size();
		int d = train_lines.get(0).split("\\s+").length - 1;
		double[][] data = new double[N][d];
		int[] labels = new int[N];
		ArrayList<String> test_lines = Reader.read_data(test);
		int test_N = test_lines.size();
		double[][] test_data = new double[test_N][d];
		int[] test_labels = new int[test_N]; 
		for(int i = 0; i <= 8; i+= 2) {
			Reader.set_one_vs_all_data(train_lines, data, labels, N, d, i);
			SVM svm = new SVM(new PolynomialKernel(2), 0.01);
			svm.train(data, labels);
			StdOut.println(String.format(java.util.Locale.UK, "%d vs all: sv number = %4d, E_in = %.5f", i, svm.sv.size(), svm.calc_error(data, labels)));
		} 
		for(int i = 1; i <= 9; i+= 2) {
			Reader.set_one_vs_all_data(train_lines, data, labels, N, d, i);
			SVM svm = new SVM(new PolynomialKernel(2), 0.01);
			svm.train(data, labels);
			StdOut.println(String.format(java.util.Locale.UK, "%d vs all: sv number = %4d, E_in = %.5f", i, svm.sv.size(), svm.calc_error(data, labels)));
		} 
	}
	
	public static void test_5_vs_1_poly(String train, String test) throws IOException {
		ArrayList<String> train_lines = Reader.read_data(train);
		ArrayList<String> test_lines = Reader.read_data(test);
		ArrayList<String> train_5_1 = new ArrayList<>();
		ArrayList<String> test_5_1 = new ArrayList<>();
		for(String s : train_lines) {
			int digit = (int) Double.parseDouble(s.split("\\s+")[0]);
			if(digit == 1 || digit == 5) {
				train_5_1.add(s);
			}
		}
		for(String s : test_lines) {
			int digit = (int) Double.parseDouble(s.split("\\s+")[0]);
			if(digit == 1 || digit == 5) {
				test_5_1.add(s);
			}
		}
		int N = train_5_1.size();
		int N_test = test_5_1.size();
		double[][] train_data = new double[N][2];
		double[][] test_data = new double[N_test][2];
		int[] train_labels = new int[N];
		int[] test_labels = new int[N_test];
		Reader.set_one_vs_one_data(train_5_1, train_data, train_labels, N, 2, 1, 5);
		Reader.set_one_vs_one_data(test_5_1, test_data, test_labels, N_test, 2, 1, 5);
		for(int i = 4; i >= 0; i--) {
			double C = Math.pow(10, -i);
			SVM svm = new SVM(new PolynomialKernel(2), C);
			svm.train(train_data, train_labels);
			StdOut.println(String.format(java.util.Locale.UK, "Q = 2, C = %.4f: SV number = %3d, E_in = %.3f, E_out = %.3f", C, svm.sv.size(), svm.calc_error(train_data, train_labels), svm.calc_error(test_data, test_labels)));
		} 
		/* for(int i = 4; i >= 0; i--) {
			double C = Math.pow(10, -i);
			SVM svm = new SVM(new PolynomialKernel(5), C);
			svm.train(train_data, train_labels);
			StdOut.println(String.format(java.util.Locale.UK, "Q = 5, C = %.4f: SV number = %3d, E_in = %.3f, E_out = %.3f", C, svm.sv.size(), svm.calc_error(train_data, train_labels), svm.calc_error(test_data, test_labels)));
		}  */
	}
	
	public static void test_5_vs_1_rbf(String train, String test) throws IOException {
		ArrayList<String> train_lines = Reader.read_data(train);
		ArrayList<String> test_lines = Reader.read_data(test);
		ArrayList<String> train_5_1 = new ArrayList<>();
		ArrayList<String> test_5_1 = new ArrayList<>();
		for(String s : train_lines) {
			int digit = (int) Double.parseDouble(s.split("\\s+")[0]);
			if(digit == 1 || digit == 5) {
				train_5_1.add(s);
			}
		}
		
		for(String s : test_lines) {
			int digit = (int) Double.parseDouble(s.split("\\s+")[0]);
			if(digit == 1 || digit == 5) {
				test_5_1.add(s);
			}
		}
		int N = train_5_1.size();
		int N_test = test_5_1.size();
		double[][] train_data = new double[N][2];
		double[][] test_data = new double[N_test][2];
		int[] train_labels = new int[N];
		int[] test_labels = new int[N_test];
		Reader.set_one_vs_one_data(train_5_1, train_data, train_labels, N, 2, 1, 5);
		Reader.set_one_vs_one_data(test_5_1, test_data, test_labels, N_test, 2, 1, 5);
		for(int i = -2; i <= 6; i += 2) {
			double C = Math.pow(10, i);
			SVM svm = new SVM(new RBFKernel(1), C);
			svm.train(train_data, train_labels);
			StdOut.println(String.format(java.util.Locale.UK, "C = %12.3f: SV number = %3d, E_in = %.7f, E_out = %.7f", C, svm.sv.size(), svm.calc_error(train_data, train_labels), svm.calc_error(test_data, test_labels)));
		} 
	}
	
	public static void set_random_data(double[][] data, int[] labels, double[] w, double b, int N, int d) {
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < d; j++) {
				data[i][j] = 2*Math.random() - 1;
			}
			labels[i] = sign(dot(w, data[i]) + b);
		}
	}
	
	public static int sign(double a) {
		if(a > 0) {
			return 1;
		}
		return -1;
	}	
	
	public static double dot(double[] x1, double[] x2) {
		double dot = 0;
		for(int i = 0; i < x1.length; i++) {
			dot += x1[i]*x2[i];
		}
		return dot;
	}
	
	public static void test_svm_vs_perceptron(int N, int T) {
		double[][] data = new double[N][2];
		int[] labels = new int[N];
		double[][] test_data = new double[10000][2];
		int[] test_labels = new int[10000];
		double[] w = new double[2];
		int sv_num = 0;
		double svm_err, perc_err;
		int svm_better = 0;
		for(int t = 0; t < T; ) {
			StdOut.println(t);
			double x1 = 2*Math.random() - 1;
			double y1 = 2*Math.random() - 1;
			double x2 = 2*Math.random() - 1;
			double y2 = 2*Math.random() - 1;
			double b = y2 - x2*(y2-y1)/(x2-x1);
			w[0] = (y2-y1)/(x2-x1);
			w[1] = -1;
			set_random_data(data, labels, w, b, N, 2);
			set_random_data(test_data, test_labels, w, b, 10000, 2);
			int pos = 0, neg = 0;
			for(int i = 0; i < N; i++) {
				if(sign(dot(w, data[i]) + b) == -1) {
					neg++;
				} else {
					pos++;
				}
			}
			if(pos == 0 || neg == 0) {
				continue;
			}
			double[][] perc_train_data = new double[N][3];
			double[][] perc_test_data = new double[10000][3];
			for(int i = 0; i < N; i++) {
				perc_train_data[i][0] = 1;
				perc_train_data[i][1] = data[i][0];
				perc_train_data[i][2] = data[i][1];
			}
			for(int i = 0; i < 10000; i++) {
				perc_test_data[i][0] = 1;
				perc_test_data[i][1] = test_data[i][0];
				perc_test_data[i][2] = test_data[i][1];
			} 
			Perceptron p = new Perceptron(perc_train_data, labels);
			SVM svm = new SVM(new LinearKernel(), Double.POSITIVE_INFINITY);
			svm.train(data, labels);
			perc_err = p.calc_error(perc_test_data, test_labels);
			svm_err = svm.calc_error(test_data, test_labels);
			if(svm_err < perc_err) {
				svm_better++;
			}
			sv_num += svm.sv.size();
			t++;
		}
		StdOut.println("\nSVM better than Perceptron in " + (svm_better+0.0)/T + " runs");
		StdOut.println("Average number of support vectors = " + (sv_num+0.0)/T);
	}	
		
	public static void main(String[] args) throws IOException {	
		Stopwatch sw = new Stopwatch();
		/* int N = Integer.parseInt(args[0]);
		int T = Integer.parseInt(args[1]);
		test_svm_vs_perceptron(N, T);  */
		test_poly(args[0], args[1]);		
		//test_5_vs_1_poly(args[0], args[1]);
		//test_5_vs_1_rbf(args[0], args[1]);
		StdOut.println("\nTiming results: " + sw.elapsedTime());
	}
}