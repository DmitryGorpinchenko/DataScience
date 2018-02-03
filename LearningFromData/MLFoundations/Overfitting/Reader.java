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
	
	public static void set_data(ArrayList<String> lines, double[][] data, int[] labels, int d, int lo, int hi) {
		String[] tokens = null;
		for(int i = lo; i < hi; i++) {
			tokens = lines.get(i).split("\\s+");
			data[i-lo][0] = 1;
			for(int j = 1; j <= d; j++) {
				data[i-lo][j] = Double.parseDouble(tokens[j-1]);
			}
			labels[i-lo] = Integer.parseInt(tokens[d]);
		}
	}
}