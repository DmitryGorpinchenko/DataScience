import java.io.*;
import java.util.ArrayList;

class Perceptron {
	
	private double[] w;
	private int d, N;
	public int iter;
	
	public Perceptron(double[][] data, int[] labels, String alg_type, double rate) throws IOException {
		N = data.length;
		d = data[0].length - 1;
		w = new double[d+1];
		switch(alg_type.split(" ")[0]) {
			case "naive": {
				train_naive(data, labels, rate); 
				break;
			}
			case "random": {
				train_random(data, labels, rate); 
				break;
			}
			case "pocket": {
				int iter_num = Integer.parseInt(alg_type.split(" ")[1]);
				train_pocket(data, labels, rate, iter_num); 
				break;
			}
			default: throw new IllegalArgumentException("Incorrect algorithm type!");
		}
	}
	
	public void train_naive(double[][] data, int[] labels, double rate) throws IOException {
		int cursor = 0;
		while(true) {
			int mistake = -1, i = cursor;
			while(true) {
				if(classify(data[i]) != labels[i]) {
					mistake = i;
					cursor = i;
					break;
				}
				i = (i + 1)%N;
				if(i == cursor) {
					break;
				}
			}
			if(mistake == -1) {
				break;
			}
			for(int j = 0; j < d+1; j++) {
				w[j] += rate*labels[mistake]*data[mistake][j];
			}
			iter++;
		}
	}
	
	public void train_random(double[][] data, int[] labels, double rate) throws IOException {
		shuffle(data, labels);
		train_naive(data, labels, rate);
	}
	
	public void train_pocket(double[][] data, int[] labels, double rate, int iter_num) throws IOException {
		double[] w_pocket = new double[d+1];
		int min_mistake_num = Integer.MAX_VALUE;
		for(int k = 0; k < iter_num; k++) {
			shuffle(data, labels);
			int mistake = -1;
			int curr_mistake_num = 0;
			for(int i = 0; i < N; i++) {
				if(classify(data[i]) != labels[i]) {
					mistake = i;
					curr_mistake_num++;
				}
			}
			if(min_mistake_num > curr_mistake_num) {
				min_mistake_num = curr_mistake_num;
				for(int j = 0; j < d+1; j++) {
					w_pocket[j] = w[j];
				}
			}
			if(mistake == -1) {
				break;
			}
			for(int j = 0; j < d+1; j++) {
				w[j] += rate*labels[mistake]*data[mistake][j];
			}
		}
		w = w_pocket;
	}
	
	public int classify(double[] x) {
		return sign(dot(w, x));
	}
	
	public double calc_error(double[][] data, int[] labels) {
		int incorrect = 0;
		for(int i = 0; i < data.length; i++) {
			if(classify(data[i]) != labels[i]) {
				incorrect++;
			}
		}
		return (incorrect + 0.0)/data.length;
	}
	
	public static void shuffle(double[][] data, int[] labels) {
		int N = data.length;
		for(int i = 0; i < N; i++) {
			int to_swap = StdRandom.uniform(i, N);
			double[] temp = data[to_swap];
			data[to_swap] = data[i];
			data[i] = temp;
			int t = labels[to_swap];
			labels[to_swap] = labels[i];
			labels[i] = t;
		}
	}
	
	public static double dot(double[] a, double[] b) {
		double dot = 0;
		for(int i = 0; i < a.length; i++) {
			dot += a[i]*b[i];
		}
		return dot;
	}
	
	public static int sign(double a) {
		if(a > 0) {
			return 1;
		}
		return -1;
	}
	
	public static ArrayList<String> read_data(String file) throws IOException {
		String line = null;
		String[] tokens = null;
		ArrayList<String> lines = new ArrayList<>();
		BufferedReader input = new BufferedReader(new FileReader(file));
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
	
	public static void test_naive(String file) throws IOException {
		ArrayList<String> lines = read_data(file);
		int N = lines.size();
		int d = lines.get(0).split("\\s+").length - 1;
		double[][] data = new double[N][d+1];
		int[] labels = new int[N];
		set_data(lines, data, labels, N, d);
		Perceptron p = new Perceptron(data, labels, "naive", 1);
		StdOut.println("Iteration to converge = " + p.iter);
	}
	
	public static void test_random(String file, double rate, int n) throws IOException {
		ArrayList<String> lines = read_data(file);
		int N = lines.size();
		int d = lines.get(0).split("\\s+").length - 1;
		double[][] data = new double[N][d+1];
		int[] labels = new int[N];
		set_data(lines, data, labels, N, d);
		int av = 0;
		for(int i = 0; i < n; i++) {
			Perceptron p = new Perceptron(data, labels, "random", rate);
			av += p.iter;
		}
		StdOut.println("Average iteration to converge = " + (av + 0.0)/n);
	}
	
	public static void test_pocket(String train_file, String test_file, String alg_type, double rate, int n) throws IOException {
		ArrayList<String> lines = read_data(train_file);
		int N = lines.size();
		int d = lines.get(0).split("\\s+").length - 1;
		double[][] data = new double[N][d+1];
		int[] labels = new int[N];
		set_data(lines, data, labels, N, d);
		lines = read_data(test_file);
		double[][] test_data = new double[N][d+1];
		int[] test_labels = new int[N];
		set_data(lines, test_data, test_labels, N, d);
		double av_error = 0;
		for(int i = 0; i < n; i++) {
			Perceptron p = new Perceptron(data, labels, alg_type, rate);
			av_error += p.calc_error(test_data, test_labels);
		}
		StdOut.println("Average out of sample error = " + (av_error + 0.0)/n);
	}
	
	public static void main(String[] args) throws IOException {
		test_naive(args[0]);
		//test_random(args[0], Double.parseDouble(args[1]), Integer.parseInt(args[2]));
		//test_pocket(args[0], args[1], args[2], Double.parseDouble(args[3]), Integer.parseInt(args[4]));
	}
}