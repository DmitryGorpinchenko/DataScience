import java.io.*;
import java.util.ArrayList;

class LogisticRegression {
	
	public static double[] get_model(double[][] data, int[] labels, int T, double eta, boolean is_stochastic) {
		int N = data.length;
		int d = data[0].length - 1;
		double[] w = new double[d+1]; //initialize all weights to zero
		if(is_stochastic) {
			run_stochastic_GD(w, data, labels, T, eta, N, d);
		} else {
			run_batch_GD(w, data, labels, T, eta, N, d);
		}	
		return w;
	}
	
	//Note: mutates w vector via addarray()
	public static void run_stochastic_GD(double[] w, double [][] data, int[] labels, int T, double eta, int N, int d) {
		int cursor = 0;
		for(int i = 0; i < T; i++) {
			addarray(w, scale(grad_err(w, data[cursor], labels[cursor]), -eta));
			cursor = (cursor+1)%N;
		}
	}
	
	//Note: mutates w vector via addarray
	public static void run_batch_GD(double[] w, double [][] data, int[] labels, int T, double eta, int N, int d) {
		for(int i = 0; i < T; i++) {
			addarray(w, scale(grad_Err(w, data, labels, N, d), -eta));
		}
	}
	
	public static double[] grad_Err(double[] w, double[][] data, int[] labels, int N, int d) {
		double[] grad = new double[d+1];
		for(int i = 0; i < N; i++) {
			addarray(grad, grad_err(w, data[i], labels[i]));
		}
		return scale(grad, 1.0/N);
	}
	
	public static double[] grad_err(double[] w, double[] x, int y) {
		return scale(x, -y*sigmoid(-y*dot(w, x)));
	}
	
	public static int classify(double[] w, double[] x) {
		return sign(dot(w, x));
	}
	
	public static double sigmoid(double x) {
		return 1.0/(1+Math.exp(-x));
	}
	
	//adds y to x and store result in x
	public static void addarray(double[] x, double[] y) {
		for(int i = 0; i < x.length; i++) {
			x[i] += y[i];
		}
	}
	
	public static double[] scale(double[] x, double a) {
		double[] scaled = new double[x.length];
		for(int i = 0 ; i < x.length; i++) {
			scaled[i] = a*x[i];
		}
		return scaled;
	}
	
	public static double dot(double[] w, double[] x) {
		double dot = 0;
		for(int i = 0; i < x.length; i++) {
			dot += w[i]*x[i];
		}
		return dot;
	}
	
	public static int sign(double x) {
		if(x > 0) {
			return 1; 
		}
		return -1;
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
	
	public static void test(String train, String test, int T, double eta, boolean is_stochastic) throws IOException {
		ArrayList<String> lines = read_data(train);
		int N = lines.size();
		int d = lines.get(0).split("\\s+").length - 1;
		double[][] data = new double[N][d+1];
		int[] labels = new int[N];
		set_data(lines, data, labels, N, d);
		double[] w = get_model(data, labels, T, eta, is_stochastic);
		lines = read_data(test);
		data = new double[N][d+1];
		labels = new int[N];
		set_data(lines, data, labels, N, d);
		int incorrect = 0;
		for(int i = 0; i < data.length; i++) {
			if(classify(w, data[i]) != labels[i]) {
				incorrect++;
			}
		}
		StdOut.println("E_out = " + (incorrect+0.0)/data.length);
	}
	
	public static void main(String[] args) throws IOException {
		int T = Integer.parseInt(args[2]);
		double eta = Double.parseDouble(args[3]);
		boolean is_stochastic = Boolean.parseBoolean(args[4]);
		test(args[0], args[1], T, eta, is_stochastic);
	}
}