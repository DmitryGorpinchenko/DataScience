import java.util.ArrayList;
import java.io.*;

class NNet {
	
	private double[][][] w;
	private int[] hidden;
	private int L;
	
	public NNet(int d, int[] hidden) {
		L = hidden.length + 1;
		w = new double[L][][];
		w[0] = new double[hidden[0]][d+1];
		for(int i = 1; i < hidden.length; i++) {
			w[i] = new double[hidden[i]][hidden[i-1]+1];
		}
		w[L-1] = new double[1][hidden[L-2]+1];
		this.hidden = hidden;
	}
	
	public void train(double[][] data, int[] labels, int T, double r, double eta) {
		init_w(r);
		int N = data.length;
		for(int t = 0; t < T; t++) {
			//choose example for the iteration
			int n = StdRandom.uniform(0, N);
			//forward prop to compute all 'extracted' features
			double[][] x = new double[L][];
			x[0] = data[n];
			for(int i = 1; i <= hidden.length; i++) {
				x[i] = new double[hidden[i-1]+1];
				x[i][0] = 1;
				for(int j = 1; j <= hidden[i-1]; j++) {
					x[i][j] = tanh(s(x[i-1], w[i-1][j-1]));
				}
			}
			//backprop
			double[][] delta = new double[L][];
			delta[L-1] = new double[1];
			double tanh = tanh(s(x[L-1], w[L-1][0]));
			delta[L-1][0] = -2*(labels[n]-tanh)*(1-tanh*tanh);
			for(int i = L-2; i >= 0; i--) {
				delta[i] = new double[hidden[i]];
				for(int j = 0; j < hidden[i]; j++) {
					double sum = 0;
					if(i == L-2) {
						sum = delta[i+1][0]*w[i+1][0][j+1];
					} else {
						for(int k = 0; k < hidden[i+1]; k++) {
							sum += delta[i+1][k]*w[i+1][k][j+1];
						}
					}
					delta[i][j] = (1-x[i+1][j+1]*x[i+1][j+1])*sum;
				}
			}
			//update weights
			for(int i = 0; i < L; i++) {
				int M = w[i].length;
				for(int j = 0; j < M; j++) {
					int K = w[i][j].length;
					for(int k = 0; k < K; k++) {
						w[i][j][k] -= eta*x[i][k]*delta[i][j];
					}
				}
			}
		}
	}
	
	public void init_w(double r) {
		for(int i = 0; i < L; i++) {
			int M = w[i].length;
			for(int j = 0; j < M; j++) {
				int K = w[i][j].length;
				for(int k = 0; k < K; k++) {
					w[i][j][k] = (2*Math.random()-1)*r;
				}
			}
		}
	}
	
	public int classify(double[] x) {
		for(int i = 0; i < L-1; i++) {
			double[] transformed_x = new double[w[i].length+1];
			transformed_x[0] = 1;
			for(int j = 1; j < transformed_x.length; j++) {
				transformed_x[j] = tanh(s(x, w[i][j-1]));
			}
			x = transformed_x;
		}
		return tanh(s(x, w[L-1][0])) > 0 ? 1:-1;
	} 
	
	public double calc_error(double[][] data, int[] labels) {
		int incorrect = 0;
		for(int i = 0; i < data.length; i++) {
			if(classify(data[i]) != labels[i]) {
				incorrect++;
			}
		}
		return (incorrect+0.0)/data.length;
	}
	
	public static double tanh(double s) {
		double exp_s = Math.exp(s), exp_minus_s = Math.exp(-s); 
		return (exp_s - exp_minus_s)/(exp_s + exp_minus_s);
	}
	
	public static double s(double[] x, double[] w) {
		double s = 0;
		for(int i = 0; i < x.length; i++) {
			s += x[i]*w[i];
		}
		return s;
	}
	
	public static ArrayList<String> read_data(String file) throws IOException {
		BufferedReader input = new BufferedReader(new FileReader(file));
		ArrayList<String> lines = new ArrayList<>();
		String line = null;
		while((line = input.readLine()) != null) {
			lines.add(line);
		}
		input.close();
		return lines;
	}
	
	public static void set_data(ArrayList<String> lines, double[][] data, int[] labels, int N, int d) {
		String[] tokens = null;
		for(int i = 0; i < N; i++) {
			tokens = lines.get(i).split("\\s+");
			data[i][0] = 1;
			for(int j = 1; j <= d; j++) {
				data[i][j] = Double.parseDouble(tokens[j-1]);
			}
			labels[i] = Integer.parseInt(tokens[d]);
		}
	}
	
	public static void q11_14(String train, String test) throws IOException {
		ArrayList<String> lines = read_data(train);
		int N = lines.size();
		int d = lines.get(0).split("\\s+").length-1;
		double[][] train_data = new double[N][d+1];
		int[] train_labels = new int[N];
		set_data(lines, train_data, train_labels, N, d);
		lines = read_data(test);
		N = lines.size();
		double[][] test_data = new double[N][d+1];
		int[] test_labels = new int[N];
		set_data(lines, test_data, test_labels, N, d);
		int T = 50000;
		StdOut.println("\n*** Question 11 ***");
		double r = 0.1, eta = 0.1;
		int[] Ms = new int[] {1, 6, 11, 16, 21};
		for(int i = 0; i < Ms.length; i++) {
			double av_E_out = 0;
			for(int j = 0; j < 500; j++) {
				NNet nnet = new NNet(d, new int[] {Ms[i]});
				nnet.train(train_data, train_labels, T, r, eta);
				av_E_out += nnet.calc_error(test_data, test_labels);
			}
			StdOut.println("M = " + Ms[i] + ": E_out = " + av_E_out/500);
		} 
		StdOut.println("\n*** Question 12 ***");
		int M = 3;
		double[] rs = new double[] {0, 0.001, 0.1, 10, 1000};
		for(int i = 0; i < rs.length; i++) {
			double av_E_out = 0;
			for(int j = 0; j < 500; j++) {
				NNet nnet = new NNet(d, new int[] {M});
				nnet.train(train_data, train_labels, T, rs[i], eta);
				av_E_out += nnet.calc_error(test_data, test_labels);
			}
			StdOut.println("r = " + rs[i] + ": E_out = " + av_E_out/500);
		}
		StdOut.println("\n*** Question 13 ***");
		double[] etas = new double[] {0.001, 0.01, 0.1, 1, 10};
		for(int i = 0; i < etas.length; i++) {
			double av_E_out = 0;
			for(int j = 0; j < 500; j++) {
				NNet nnet = new NNet(d, new int[] {M});
				nnet.train(train_data, train_labels, T, r, etas[i]);
				av_E_out += nnet.calc_error(test_data, test_labels);
			}
			StdOut.println("eta = " + etas[i] + ": E_out = " + av_E_out/500);
		} 
		StdOut.println("\n*** Question 14 ***");
		eta = 0.01;
		double av_E_out = 0;
		for(int j = 0; j < 500; j++) {
			NNet nnet = new NNet(d, new int[] {8, 3});
			nnet.train(train_data, train_labels, T, r, eta);
			av_E_out += nnet.calc_error(test_data, test_labels);
		}
		StdOut.println("d-8-3-1 NNet: E_out = " + av_E_out/500);
	}
	
	public static void main(String[] args) throws IOException {
		Stopwatch sw = new Stopwatch();
		q11_14(args[0], args[1]);
		StdOut.println("Timing results: " + sw.elapsedTime());
	}
}