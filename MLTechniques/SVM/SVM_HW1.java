import java.util.ArrayList;
import java.io.*;

class SVM_HW1 {
	
	public static void q3_4() {
		StdOut.println("*** Question 3 ***\n");
		double[][] data = new double[][] {{1, 0}, {0, 1}, {0, -1}, {-1, 0}, {0, 2}, {0, -2}, {-2, 0}};
		int[] labels = new int[] {-1, -1, -1, 1, 1, 1, 1};
		SVM svm = new SVM(new PolynomialKernel(2), Double.POSITIVE_INFINITY);
		svm.train(data, labels);
		double sum = 0;
		for(int i = 0; i < labels.length; i++) {
			sum += svm.alpha[i]/labels[i];
			StdOut.print(String.format(java.util.Locale.UK, "%.3f ", svm.alpha[i]/labels[i]));
		}
		StdOut.println(String.format(java.util.Locale.UK, "\nb = %.3f ", svm.b));
		StdOut.println(String.format(java.util.Locale.UK, "sum = %.4f", sum));
		StdOut.println("\n*** Question 4 ***\n");
		StdOut.println("   SVM  curve1 curve2 curve3 curve4");
		for(int i = 0; i < labels.length; i++) {
			StdOut.println(String.format(java.util.Locale.UK, "\n%6.3f %6.3f %6.3f %6.3f %6.3f", 
			                             svm.f(data[i]) + svm.b, f_q4(0, -16.0/9, 6.0/9, 8.0/9, 15.0/9, data[i]), f_q4(0, -16.0/9, 6.0/9, 8.0/9, -15.0/9, data[i]), 
			                                                     f_q4(-16.0/9, 0, 8.0/9, 6.0/9, -15.0/9, data[i]), f_q4(-16.0/9, 0, 8.0/9, 6.0/9, 15.0/9, data[i])));
		}
	}
	
	public static double f_q4(double k1, double k2, double k12, double k22, double k0, double[] x) {
		return k1*x[0] + k2*x[1] + k12*x[0]*x[0] + k22*x[1]*x[1] + k0;
	} 
	
	public static void q15(String file) throws IOException {
		StdOut.println("\n*** Question 15 ***\n");
		ArrayList<String> lines = Reader.read_data(file);
		int N = lines.size();
		int d = 2;
		double[][] data = new double[N][d]; 
		int[] labels = new int[N];
		Reader.set_one_vs_all_data(lines, data, labels, N, d, 0);
		SVM svm = new SVM(new LinearKernel(), 0.01);
		double[][] Q = svm.kernel.transform(data, labels); 
		int[] id = new int[N];
		for(int i = 0; i < N; i++) {
			id[i] = i;
		}
		svm.train(Q, data, labels, id);
		StdOut.println(String.format(java.util.Locale.UK, "\n||w|| = %.3f ", svm.abs_w()));
	}
		
	public static void q16_17(String file) throws IOException {
		StdOut.println("\n*** Question 16 and 17 ***\n");
		ArrayList<String> lines = Reader.read_data(file);
		int N = lines.size();
		int d = 2;
		double[][] data = new double[N][d]; 
		int[] labels = new int[N];
		for(int i = 0; i <= 8; i += 2) {
			Reader.set_one_vs_all_data(lines, data, labels, N, d, i);
			SVM svm = new SVM(new PolynomialKernel(2), 0.01);
			double[][] Q = svm.kernel.transform(data, labels); 
			int[] id = new int[N];
			for(int k = 0; k < N; k++) {
				id[k] = k;
			}
			svm.train(Q, data, labels, id);
			double sum = 0;
			for(int j = 0; j < N; j++) {
				sum += svm.alpha[j]/labels[id[j]];
			}
			StdOut.println(String.format(java.util.Locale.UK,"%d vs all E_in = %.4f, sum_aplpha = %.4f", i, svm.calc_error(data, labels), sum));
		}
	}
	
	public static void q18(String train_file, String test_file) throws IOException {
		StdOut.println("\n*** Question 18 ***\n");
		ArrayList<String> train_lines = Reader.read_data(train_file);
		ArrayList<String> test_lines = Reader.read_data(test_file);
		int N = train_lines.size();
		int d = 2;
		double[][] train_data = new double[N][d]; 
		int[] train_labels = new int[N];
		double[][] test_data = new double[test_lines.size()][d];
		int[] test_labels = new int[test_lines.size()];
		Reader.set_one_vs_all_data(train_lines, train_data, train_labels, N, d, 0);
		Reader.set_one_vs_all_data(test_lines, test_data, test_labels, test_lines.size(), d, 0);
		Kernel kernel = new RBFKernel(100);
		double[][] Q = kernel.transform(train_data, train_labels); 
		int[] id = new int[N];
		for(int k = 0; k < N; k++) {
			id[k] = k;
		}
		for(double C = 0.001; C <= 10; C *= 10) {
			SVM svm = new SVM(kernel, C); 
			svm.train(Q, train_data, train_labels, id);
			double E_out = svm.calc_error(test_data, test_labels);
			int sv_num = svm.sv.size();
			double dist = 1/svm.abs_w();
			double obj = svm.obj();
			double ksi_sum = ksi_sum(svm, train_data, train_labels, id);
			StdOut.println(String.format(java.util.Locale.UK, "\nC = %6.3f: E_out = %.3f, sv_num = %4d, dist = %.4f, obj = %.3f, ksi_sum = %.3f", C, E_out, sv_num, dist, obj, ksi_sum));
		}
	}
	
	public static double ksi_sum(SVM svm, double[][] data, int[] labels, int[] id) {
		double sum = 0;
		for(int i : svm.sv.keySet()) {
			if(Math.abs(svm.alpha[i]) == svm.C) {
				sum += (1 - labels[id[i]]*(svm.f(data[id[i]]) + svm.b));
			}
		}
		return sum;
	}	
	
	public static void q19(String train_file, String test_file) throws IOException {
		StdOut.println("\n*** Question 19 ***\n");
		ArrayList<String> train_lines = Reader.read_data(train_file);
		ArrayList<String> test_lines = Reader.read_data(test_file);
		int N = train_lines.size();
		int d = 2;
		double[][] train_data = new double[N][d]; 
		int[] train_labels = new int[N];
		double[][] test_data = new double[test_lines.size()][d];
		int[] test_labels = new int[test_lines.size()];
		Reader.set_one_vs_all_data(train_lines, train_data, train_labels, N, d, 0);
		Reader.set_one_vs_all_data(test_lines, test_data, test_labels, test_lines.size(), d, 0);
		for(int gamma = 1; gamma <= 10000; gamma *= 10) {
			SVM svm = new SVM(new RBFKernel(gamma), 0.1);
			double[][] Q = svm.kernel.transform(train_data, train_labels); 
			int[] id = new int[N];
			for(int k = 0; k < N; k++) {
				id[k] = k;
			}
			svm.train(Q, train_data, train_labels, id);
			StdOut.println(String.format(java.util.Locale.UK, "\ngamma = %-5d: E_out = %.3f ", gamma, svm.calc_error(test_data, test_labels)));
		}
	}
	
	public static void q20(String file) throws IOException {
		StdOut.println("\n*** Question 20 ***\n");
		ArrayList<String> lines = Reader.read_data(file);
		int N = lines.size();
		int d = 2;
		double[][] data = new double[N][d]; 
		int[] labels = new int[N];
		int[][] id = new int[100][N];
		for(int i = 0; i < 100; i++) {
			for(int j = 0; j < N; j++) {
				id[i][j] = j;
			}
			shuffle(id[i]);
		}
		int[] train_id = new int[N-1000];
		Reader.set_one_vs_all_data(lines, data, labels, N, d, 0); 
		double[][] val_data = new double[1000][d]; 
		int[] val_labels = new int[1000];
		int[] freq = new int[5]; 
		double[][] E_val = new double[5][100];
		for(int i = 0; i <= 4; i++) {	                       
			double gamma = Math.pow(10, i);
			StdOut.println("\ngamma = " + gamma);
			SVM svm = new SVM(new RBFKernel(gamma), 0.1);
			double[][] Q = svm.kernel.transform(data, labels);
			for(int t = 0; t < 100; t++) {
				StdOut.print(t + " ");
				for(int j = 0; j < 1000; j++) {
					val_data[j] = data[id[t][j]];
					val_labels[j] = labels[id[t][j]];
				}
				for(int j = 1000; j < N; j++) {					
					train_id[j-1000] = id[t][j]; 
				}
				svm.train(Q, data, labels, train_id);
				E_val[i][t] = svm.calc_error(val_data, val_labels);
			}
		}
		StdOut.println();
		for(int t = 0; t < 100; t++) {
			double min_E_val = Double.POSITIVE_INFINITY;
			int selected_gamma = -1;
			for(int i = 0; i <= 4; i++) {        
				if(min_E_val > E_val[i][t]) {
					min_E_val = E_val[i][t];
					selected_gamma = i;
				}
			}
			freq[selected_gamma]++; 
		}
		for(int i = 0; i < 5; i++) {
			StdOut.println(String.format(java.util.Locale.UK, "%.1f selected %d times", Math.pow(10, i), freq[i]));
		}
	}
	
	public static void shuffle(int[] id) {
		int N = id.length;
		for(int i = 0; i < N; i++) {
			int to_swap = StdRandom.uniform(i, N);
			int t = id[to_swap];
			id[to_swap] = id[i];
			id[i] = t;
		}
	}
		
	public static void main(String[] args) throws IOException {
		Stopwatch sw = new Stopwatch();
		//q3_4();
		//q15(args[0]);
		//q16_17(args[0]);
		//q18(args[0], args[1]);
		//q19(args[0], args[1]);
		q20(args[0]);
		StdOut.println(String.format(java.util.Locale.UK, "\nTiming results: %.4f", sw.elapsedTime()));
	}
}