import java.util.ArrayList;
import java.io.*;

class HW3 {

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
	
	public static void q13_14_15(String train_file, String test_file) throws IOException {
		ArrayList<String> lines = read_data(train_file);
		int N = lines.size();
		int d = 2;
		double[][] data = new double[N][d];
		int[] labels = new int[N];
		for(int i = 0; i < N; i++) {
			String[] tokens = lines.get(i).split(" ");
			for(int j = 0; j < d; j++) {
				data[i][j] = Double.parseDouble(tokens[j]);
			}
			labels[i] = Integer.parseInt(tokens[d]);
		}
		lines = read_data(test_file);
		N = lines.size();
		double[][] test_data = new double[N][d];
		int[] test_labels = new int[N];
		for(int i = 0; i < N; i++) {
			String[] tokens = lines.get(i).split(" ");
			for(int j = 0; j < d; j++) {
				test_data[i][j] = Double.parseDouble(tokens[j]);
			}
			test_labels[i] = Integer.parseInt(tokens[d]);
		}
		DecisionTree dt = new DecisionTree("CART", Integer.MAX_VALUE);
		dt.train(data, labels);
		StdOut.println("Q13: number of internal nodes = " + dt.get_node_count());
		double E_in = dt.calc_error(data, labels);
		StdOut.println("Q14: E_in = " + E_in);
		double E_out = dt.calc_error(test_data, test_labels);
		StdOut.println("Q15: E_out = " + E_out);
	}
	
	public static void q16_17_18(String train_file, String test_file) throws IOException {
		ArrayList<String> lines = read_data(train_file);
		int N = lines.size();
		int d = 2;
		double[][] data = new double[N][d];
		int[] labels = new int[N];
		for(int i = 0; i < N; i++) {
			String[] tokens = lines.get(i).split(" ");
			for(int j = 0; j < d; j++) {
				data[i][j] = Double.parseDouble(tokens[j]);
			}
			labels[i] = Integer.parseInt(tokens[d]);
		}
		lines = read_data(test_file);
		N = lines.size();
		double[][] test_data = new double[N][d];
		int[] test_labels = new int[N];
		for(int i = 0; i < N; i++) {
			String[] tokens = lines.get(i).split(" ");
			for(int j = 0; j < d; j++) {
				test_data[i][j] = Double.parseDouble(tokens[j]);
			}
			test_labels[i] = Integer.parseInt(tokens[d]);
		}
		int T = 300, trials = 100;
		double av_E_in_g = 0, av_E_in_G = 0, av_E_out_G = 0;
		for(int i = 0; i < trials; i++) {
			RandomForest rf = new RandomForest(T, Integer.MAX_VALUE);
			rf.train(data, labels);
			for(int t = 0; t < T; t++) {
				av_E_in_g += rf.calc_error(data, labels, t);
			}
			av_E_in_G += rf.calc_error(data, labels);
			av_E_out_G += rf.calc_error(test_data, test_labels);
		}
		av_E_in_g /= (trials*T); av_E_in_G /= trials; av_E_out_G /= trials;
		StdOut.println("Q16: av_E_in_g = " + av_E_in_g);
		StdOut.println("Q17: av_E_in_G = " + av_E_in_G);
		StdOut.println("Q18: av_E_out_G = " + av_E_out_G);
	}

	public static void q19_20(String train_file, String test_file) throws IOException {
		ArrayList<String> lines = read_data(train_file);
		int N = lines.size();
		int d = 2;
		double[][] data = new double[N][d];
		int[] labels = new int[N];
		for(int i = 0; i < N; i++) {
			String[] tokens = lines.get(i).split(" ");
			for(int j = 0; j < d; j++) {
				data[i][j] = Double.parseDouble(tokens[j]);
			}
			labels[i] = Integer.parseInt(tokens[d]);
		}
		lines = read_data(test_file);
		N = lines.size();
		double[][] test_data = new double[N][d];
		int[] test_labels = new int[N];
		for(int i = 0; i < N; i++) {
			String[] tokens = lines.get(i).split(" ");
			for(int j = 0; j < d; j++) {
				test_data[i][j] = Double.parseDouble(tokens[j]);
			}
			test_labels[i] = Integer.parseInt(tokens[d]);
		}
		int T = 300, trials = 100;
		double av_E_in_G = 0, av_E_out_G = 0;
		for(int i = 0; i < trials; i++) {
			RandomForest rf = new RandomForest(T, 1);
			rf.train(data, labels);
			av_E_in_G += rf.calc_error(data, labels);
			av_E_out_G += rf.calc_error(test_data, test_labels);
		}
		av_E_in_G /= trials; av_E_out_G /= trials;
		StdOut.println("Q19: av_E_in_G = " + av_E_in_G);
		StdOut.println("Q20: av_E_out_G = " + av_E_out_G);
	}
	
	public static void main(String[] args) throws IOException {
		q13_14_15(args[0], args[1]);
		//q16_17_18(args[0], args[1]);
		//q19_20(args[0], args[1]);
	} 	
}