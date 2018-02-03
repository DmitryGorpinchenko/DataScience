import java.util.HashSet;
import java.util.ArrayList;
import java.io.*;

class kMeans {
	
	private int k;
	private double[][] centers;
	private HashSet<double[]>[] clusters;
	private int N, d;
	
	public kMeans(int k) {
		this.k = k;
		centers = new double[k][];
		clusters = (HashSet<double[]>[]) new HashSet[k];
		for(int i = 0; i < k; i++) {
			clusters[i] = new HashSet<double[]>();
		}
	}
	
	public void cluster(double[][] data) {
		N = data.length;
		d = data[0].length;
		int[] map = new int[N];
		for(int i = 0; i < N; i++) {
			map[i] = -1;
		}
		init_centers(data);
		while(true) {
			boolean is_changed = false;
			for(int i = 0; i < N; i++) {
				int new_center = find_closest_center(data[i]);
				int old_center = map[i];
				if(new_center != old_center) {
					is_changed = true;
					map[i] = new_center;
					if(old_center != -1) {
						clusters[old_center].remove(data[i]);
					}
					clusters[new_center].add(data[i]);
				}
			}
			if(!is_changed) {
				break;
			}
			optimize_centers();
		}
	}
	
	private void optimize_centers() {
		for(int i = 0; i < k; i++) {
			double[] center = new double[d];
			for(int j = 0; j < d; j++) {
				for(double[] x : clusters[i]) {
					center[j] += x[j];
				}
				center[j] /= clusters[i].size();
			}
			centers[i] = center;
		}
	}
	
	private int find_closest_center(double[] x) {
		int id = -1;
		double min_dist = Double.POSITIVE_INFINITY;
		for(int i = 0; i < k; i++) {
			double dist = dist(centers[i], x);
			if(dist < min_dist) {
				min_dist = dist;
				id = i;
			}
		}
		return id;
	}
	
	private void init_centers(double[][] data) {
		int[] ids = new int[N];
		for(int i = 0; i < N; i++) {
			ids[i] = i;
		}
		for(int i = 0; i < k; i++) {
			int id = StdRandom.uniform(i, N);
			int temp = ids[id];
			ids[id] = ids[i];
			ids[i] = temp;
		}
		for(int i = 0; i < k; i++) {
			centers[i] = data[ids[i]];
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
	
	public static void set_data(ArrayList<String> lines, double[][] data, int N, int d) {
		String[] tokens = null;
		for(int i = 0; i < N; i++) {
			tokens = lines.get(i).split("\\s+");
			for(int j = 0; j < d; j++) {
				data[i][j] = Double.parseDouble(tokens[j]);
			}
		}
	}
	
	public double calc_error() {
		double error = 0;
		for(int i = 0; i < k; i++) {
			for(double[] x : clusters[i]) {
				error += dist(x, centers[i]);
			}
		}
		return error/N;
	}
	
	public static void q19_20(String train) throws IOException {
		ArrayList<String> lines = read_data(train);
		int N = lines.size();
		int d = lines.get(0).split("\\s+").length;
		double[][] train_data = new double[N][d];
		set_data(lines, train_data, N, d);
		double av_E_in = 0;
		for(int i = 0; i < 500; i++) {
			kMeans kmeans = new kMeans(2);
			kmeans.cluster(train_data);
			av_E_in += kmeans.calc_error();
		}
		StdOut.println("k = 2: av_E_in = " + av_E_in/500);
		av_E_in = 0;
		for(int i = 0; i < 500; i++) {
			kMeans kmeans = new kMeans(10);
			kmeans.cluster(train_data);
			av_E_in += kmeans.calc_error();
		}
		StdOut.println("k = 10: av_E_in = " + av_E_in/500);
	}
	
	public static void main(String[] args) throws IOException {
		q19_20(args[0]);// run tests
	}
}