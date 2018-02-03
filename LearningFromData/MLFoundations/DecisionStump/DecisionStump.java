import java.util.Arrays;
import java.util.ArrayList;
import java.io.*;

class DecisionStump {
	
	private int s;
	private double theta = -1;
	private int decision_feature;
	private int N;
	public double E_in = -1;
	public double E_out = -1;
	
	//assumes sorted data and labels as parallel arrays
	public DecisionStump(double[] data, int[] labels) {
		N = data.length;
		train(data, labels);
	}
	
	//data arrays of size (d x N) where d is a number of dimentions, N is the number of data points
	public DecisionStump(double[][] data, int[] labels) {
		N = data[0].length;
		int d = data.length;
		double best_E_in = Double.POSITIVE_INFINITY;
		double best_E_out = -1;
		double best_theta = -1;
		int best_s = -1;
		for(int i = 0; i < d; i++) {
			sort(data, labels, i); //inefficient sorting, just for relatively small data sets
			train(data[i], labels);
			if(E_in < best_E_in) {
				best_E_in = E_in;
				best_E_out = E_out;
				best_theta = theta;
				best_s = s;
				decision_feature = i;
			}
		}
		E_in = best_E_in;
		E_out = best_E_out;
		theta = best_theta;
		s = best_s;
	}
	
	public void train(double[] data, int[] labels) {
		int pos = 0;
		int neg = 0;
		for(int i = 0; i < N; i++) {
			if(labels[i] == 1) {
				pos++;
			} else {
				neg++;
			}
		}
		int curr_error = -1;
		if(pos > neg) {
			s = 1;
			curr_error = neg;
		} else {
			s = -1;
			curr_error = pos;
		} 
		int left_pos = 0;
		int left_neg = 0;
		for(int i = 0; i < N-1; i++) {
			double prev = data[i]; 
			double curr = data[i+1];
			double mid = prev + (curr - prev)/2;
			if(labels[i] == 1) {
				left_pos++;
			} else {
				left_neg++;
			}	
			int curr_pos_error = left_pos + (neg - left_neg);
			int curr_neg_error = left_neg + (pos - left_pos);
			//randomization on whether accept or reject an update in case of ties
			if(curr_pos_error < curr_error /*|| (curr_pos_error == curr_error && Math.random() < 0.5)*/) {
				theta = mid;
				s = 1;
				curr_error = curr_pos_error;
			}
			if(curr_neg_error < curr_error /*|| (curr_neg_error == curr_error && Math.random() < 0.5)*/) {
				theta = mid;
				s = -1;
				curr_error = curr_neg_error;
			}
		}
		E_in = (curr_error+0.0)/N;
		E_out = 0.5 + 0.3*s*(Math.abs(theta) - 1);
	}
	
	public int classify(double x) {
		return s*sign(x - theta);
	}
	
	public double calc_error(double[][] data, int[] labels) {
		int incorrect = 0;
		int N = data[0].length;
		for(int i = 0; i < N; i++) {
			if(classify(data[decision_feature][i]) != labels[i]) {
				incorrect++;
			}
		}
		return (incorrect + 0.0)/N;
	}
	
	public static int sign(double a) {
		if(a > 0) {
			return 1;
		}
		return -1;
	}
	
	//sort columns and corresponding labels by value in row 'id'
	public static void sort(double[][] data, int[] labels, int id) {
		int N = data[0].length;
		int d = data.length;
		for(int i = 0; i < N; i++) {
			for(int j = i; j > 0; j--) {
				if(data[id][j] < data[id][j-1]) {
					for(int k  = 0; k < d; k++) {
						double temp = data[k][j];
						data[k][j] = data[k][j-1];
						data[k][j-1] = temp;
					}
					int t = labels[j];
					labels[j] = labels[j-1];
					labels[j-1] = t;
				} else {
					break;
				}
			}
		}
	}
	
	public static void set_sorted_random_data(double[] x, int N) {
		for(int i = 0; i < N; i++) {
			x[i] = 2*Math.random() - 1;
		}
		Arrays.sort(x);
	}
	
	public static void set_labels(double[] data, int[] labels, int N) {
		for(int i = 0; i < N; i++) {
			labels[i] = sign(data[i]);
			if(Math.random() < 0.2) {
				labels[i] = -labels[i];
			}
		}
	}
	
	public static void test_1D_stump(int N, int T) {
		double E_in = 0;
		double E_out = 0;
		for(int i = 0; i < T; i++) {
			double[] data = new double[N];
			int[] labels = new int[N];
			set_sorted_random_data(data, N);
			set_labels(data, labels, N);
			DecisionStump ds = new DecisionStump(data, labels);
			E_in += ds.E_in;
			E_out += ds.E_out;
		}
		StdOut.println("E_in = " + E_in/T);
		StdOut.println("E_out = " + E_out/T);
	}
	
	public static void test_multiD_stump(String train, String test) throws IOException {
		String line = null;
		String[] tokens = null;
		BufferedReader input = new BufferedReader(new FileReader(train));
		ArrayList<String> lines = new ArrayList<>();
		while((line = input.readLine()) != null) {
			lines.add(line);
		}
		input.close();
		int N = lines.size();
		int d = lines.get(0).split("\\s+").length - 1;
		double[][] data = new double[d][N];
		int[] labels = new int[N];
		for(int i = 0; i < N; i++) {
			tokens = lines.get(i).split("\\s+");
			for(int j = 0; j < d; j++) {
				data[j][i] = Double.parseDouble(tokens[j]);
			}
			labels[i] = Integer.parseInt(tokens[d]);
		}
		DecisionStump ds = new DecisionStump(data, labels);
		StdOut.println("E_in = " + ds.E_in);
		input = new BufferedReader(new FileReader(test));
		lines = new ArrayList<>();
		while((line = input.readLine()) != null) {
			lines.add(line);
		}
		input.close();
		N = lines.size();
		d = lines.get(0).split("\\s+").length - 1;
		data = new double[d][N];
		labels = new int[N];
		for(int i = 0; i < N; i++) {
			tokens = lines.get(i).split("\\s+");
			for(int j = 0; j < d; j++) {
				data[j][i] = Double.parseDouble(tokens[j]);
			}
			labels[i] = Integer.parseInt(tokens[d]);
		}
		StdOut.println("E_out = " + ds.calc_error(data, labels));
	}
	
	public static void main(String[] args) throws IOException {
		//int N = Integer.parseInt(args[0]);
		//int T = Integer.parseInt(args[1]);
		//test_1D_stump(N, T);
		 test_multiD_stump(args[0], args[1]);
	}
}