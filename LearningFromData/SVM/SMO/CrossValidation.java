import java.io.*;
import java.util.ArrayList;

class CrossValidation {

	public static void cross_validation(String train) throws IOException {
		//set training data
		ArrayList<String> train_lines = Reader.read_data(train);
		ArrayList<String> train_5_1 = new ArrayList<>();
		for(String s : train_lines) {
			int digit = (int) Double.parseDouble(s.split("\\s+")[0]);
			if(digit == 1 || digit == 5) {
				train_5_1.add(s);
			}
		}
		int N = train_5_1.size();
		double[][] data = new double[N][2];
		int[] labels = new int[N];
		Reader.set_one_vs_one_data(train_5_1, data, labels, N, 2, 1, 5);
		//apply cross validation
		int[] counts = new int[5];
		double[] av_E_cv = new double[5];
		for(int t = 0; t < 100; t++) {
			StdOut.println(t + " run in progress ...");
			shuffle(data, labels);
			double best_E_cv = Double.POSITIVE_INFINITY;
			int best_k = -1;
			for(int k = 4; k >= 0; k--) {
				double C = Math.pow(10, -k);
				double E_cv = 0;
				for(int f = 0; f < 10; f++) {
					int train_size = N - 156, val_size = 156;
					int val_begin = 156*f, val_end = val_begin + 156;
					if(f == 9) {
						train_size = N - 157;
						val_size = 157;
						val_end = val_begin + 157;
					}
					double[][] train_data = new double[train_size][2];
					int[] train_labels = new int[train_size];
					double[][] val_data = new double[val_size][2];
					int[] val_labels = new int[val_size];
					for(int i = 0, train_count = 0, val_count = 0; i < N; i++) {
						if(i >= val_begin && i < val_end) {
							val_data[val_count] = data[i];
							val_labels[val_count] = labels[i];
							val_count++;
						} else {
							train_data[train_count] = data[i];
							train_labels[train_count] = labels[i];
							train_count++;
						}
					}
					SVM svm = new SVM(new PolynomialKernel(2), C);
					svm.train(train_data, train_labels);
					E_cv += svm.calc_error(val_data, val_labels);
				}
				E_cv /= 10;
				if(best_E_cv > E_cv) {
					best_E_cv = E_cv;
					best_k = k;
				}
			}
			counts[best_k]++;
			av_E_cv[best_k] += best_E_cv;
		}
		//calc average cross validation error
		for(int i = 0; i < 5; i++) {
			av_E_cv[i] /= counts[i];
		}
		//find the best model parameter C
		StdOut.println("\n*** Results ***\n");
		int id = -1;
		int max = Integer.MIN_VALUE;
		for(int i = 4; i >= 0; i--) {
			StdOut.println(String.format(java.util.Locale.UK, "C = %.5f, count = %2d, E_cv = %.7f", Math.pow(10, -i), counts[i], av_E_cv[i]));
			if(counts[i] > max) {
				max = counts[i];
				id = i;
			}
		}
		StdOut.println("\nThe winning model is C = " + Math.pow(10, -id));
		StdOut.println("Average E_cv = " + av_E_cv[id]);
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

	public static void main(String[] args) throws IOException {
		Stopwatch sw = new Stopwatch();
		cross_validation(args[0]);
		StdOut.println("\nTiming results: " + sw.elapsedTime());
	}
}