import java.io.*;
import java.util.ArrayList;

class kNN {
	
	private int k;
	private double[][] data;
	private int[] labels;
	
	public kNN(int k) {
		this.k = k;
	}
	
	public void train(double[][] data, int[] labels) {
		this.data = data;
		this.labels = labels;
	}
	
	public int classify(double[] x) {
		double[] distances = new double[data.length];
		int[] ids = new int[data.length];
		for(int i = 0; i < data.length; i++) {
			double dist = dist(data[i], x);
			distances[i] = dist;
			ids[i] = i;
		}
		sort(distances, ids);
		int pos_count = 0, neg_count = 0;
		for(int i = 0; i < k; i++) {
			if(labels[ids[i]] == 1) {
				pos_count++;
			} else {
				neg_count++;
			}
		}
		return pos_count > neg_count ? 1:-1;
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
	
	public static void sort(double[] distances, int[] ids) {
		int N = distances.length;
		for(int i = 0; i < N; i++) {
			for(int j = i; j > 0; j--) {
				if(distances[j] < distances[j-1]) {
					double temp = distances[j];
					distances[j] = distances[j-1];
					distances[j-1] = temp;
					int t = ids[j];
					ids[j] = ids[j-1];
					ids[j-1] = t;
				} else {
					break;
				}
			}
		}
	}
	
	private static double dist(double[] a, double[] b) {
		double dist = 0;
		for(int i = 0; i < a.length; i++) {
			dist += (a[i] - b[i])*(a[i] - b[i]);
		}
		return dist;
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
			for(int j = 0; j < d; j++) {
				data[i][j] = Double.parseDouble(tokens[j]);
			}
			labels[i] = Integer.parseInt(tokens[d]);
		}
	}
	
	public static void q15_18(String train, String test) throws IOException {
		ArrayList<String> lines = read_data(train);
		int N = lines.size();
		int d = lines.get(0).split("\\s+").length-1;
		double[][] train_data = new double[N][d];
		int[] train_labels = new int[N];
		set_data(lines, train_data, train_labels, N, d);
		lines = read_data(test);
		N = lines.size();
		double[][] test_data = new double[N][d];
		int[] test_labels = new int[N];
		set_data(lines, test_data, test_labels, N, d);
		kNN knn = new kNN(1);
		knn.train(train_data, train_labels);
		double E_in = knn.calc_error(train_data, train_labels);
		double E_out = knn.calc_error(test_data, test_labels);
		StdOut.println("1NN results: E_in = " + E_in + ", E_out = " + E_out);
		knn = new kNN(5);
		knn.train(train_data, train_labels);
		E_in = knn.calc_error(train_data, train_labels);
		E_out = knn.calc_error(test_data, test_labels);
		StdOut.println("1NN results: E_in = " + E_in + ", E_out = " + E_out);
	}
	
	public static void main(String[] args) throws IOException {
		q15_18(args[0], args[1]);
	}
}