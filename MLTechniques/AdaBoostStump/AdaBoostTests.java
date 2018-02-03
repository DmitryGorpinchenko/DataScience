import java.util.ArrayList;
import java.io.*;

class AdaBoostTests {
	
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
	
	public static void q12_18(String train_file, String test_file) throws IOException {
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
		int T = 300;
		AdaBoostStump abs = new AdaBoostStump(T);
		abs.train(data, labels);
		double E_in_1 = abs.stumps[0].E_in;
		double E_in = abs.calc_error(data, labels);
		double E_out_1 = abs.stumps[0].calc_error(test_data, test_labels);
		double E_out = abs.calc_error(test_data, test_labels);
		double min_eps = Double.POSITIVE_INFINITY;
		for(int t = 0; t < T; t++) {
			double eps = abs.stumps[t].weighted_error/abs.weight_sums[t];
			if(eps < min_eps) {
				min_eps = eps;
			}
		}
		StdOut.println("E_in(g_1) = " + E_in_1);
		StdOut.println("E_in(G) = " + E_in);
		StdOut.println("U_2 = " + abs.weight_sums[1]);
		StdOut.println("U_T = " + abs.weight_sums[abs.T-1]);
		StdOut.println("min(e_t) = " + min_eps);
		StdOut.println("E_out(g_1) = " + E_out_1);
		StdOut.println("E_out(G) = " + E_out);
	}
	
	public static void main(String[] args) throws IOException {
		q12_18(args[0], args[1]);
	}
}