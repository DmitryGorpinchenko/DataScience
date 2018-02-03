import java.io.*;
import java.util.ArrayList;

class Reader {

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
	
	public static void set_one_vs_all_data(ArrayList<String> lines, double[][] data, int[] labels, int N, int d, int one) {
		String[] tokens = null;
		for(int i = 0; i < N; i++) {
			tokens = lines.get(i).split("\\s+");
			for(int j = 0; j < d; j++) {
				data[i][j] = Double.parseDouble(tokens[j+1]);
			}
			int digit = (int) Double.parseDouble(tokens[0]);
			if(digit == one) {
				labels[i] = 1;
			} else {
				labels[i] = -1;
			}
		}
	}
	
	public static void set_one_vs_one_data(ArrayList<String> lines, double[][] data, int[] labels, int N, int d, int one, int two) {
		String[] tokens = null;
		for(int i = 0; i < N; i++) {
			tokens = lines.get(i).split("\\s+");
			for(int j = 0; j < d; j++) {
				data[i][j] = Double.parseDouble(tokens[j+1]);
			}
			int digit = (int) Double.parseDouble(tokens[0]);
			if(digit == one) {
				labels[i] = 1;
			} else if(digit == two) {
				labels[i] = -1;
			}
		}
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
}