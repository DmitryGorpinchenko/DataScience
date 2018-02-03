import java.io.*;
import java.util.ArrayList;

class LinearRegression {

	public static Model get_model(double[][] train_data, int[] labels) {
		SimpleMatrix X = new SimpleMatrix(train_data);
		SimpleMatrix pinvX = X.pinv();
		return new LinearDecisionBoundary(pinvX.multiply(labels));
	}
	
	public static Model get_model(double[][] train_data, int[] labels, double lambda) {
		SimpleMatrix X = new SimpleMatrix(train_data);
		SimpleMatrix X_prime = X.transpose();
		SimpleMatrix reg_pinvX = X_prime.multiply(X).plus(lambda).pinv().multiply(X_prime);
		return new LinearDecisionBoundary(reg_pinvX.multiply(labels));
	}
	
	public static ArrayList<String> read_dataset(String data_file) throws IOException {
		BufferedReader input = new BufferedReader(new FileReader(data_file));
		String line = null;
		ArrayList<String> lines = new ArrayList<>();
		while(true) {
			line = input.readLine();
			if(line == null) {
				break;
			}
			lines.add(line);
		}
		input.close();
		return lines;
	}
	
	public static double[][] get_dataset(int N) {
		double[][] training_set = new double[N][3];
		for(int i = 0; i < N; i++) {
			double x = 2*Math.random() - 1;
			double y = 2*Math.random() - 1;
			training_set[i][0] = 1; //artificial feature to be consistent with model formalism
			training_set[i][1] = x;
			training_set[i][2] = y;
		}
		return training_set;
	}
	
	public static int[] get_labels(double[][] dataset, Model model) {
		int N = dataset.length;
		int[] labels = new int[N];
		for(int i = 0; i < N; i++) {
			labels[i] = model.classify(dataset[i]);
		}
		return labels;
	}
	
	public static void add_noise(int[] labels, double p) {
		int N = labels.length;
		for(int i = 0; i < N; i++) {
			if(Math.random() <= p) {
				labels[i] = -labels[i];
			}
		}
	}
	
	public static double calc_error(Model regression_model, double[][] dataset, int[] labels) {
		int N = dataset.length;
		int incorrect = 0;
		for(int i = 0; i < N; i++) {
			if(regression_model.classify(dataset[i]) != labels[i]) {
				incorrect++;
			}	
		} 
		return (incorrect + 0.0)/N; 
	}
	
	public static Model[] get_hypotheses() {
		Model[] hypotheses = new QuadraticDecisionBoundary[5];
		hypotheses[0] = new QuadraticDecisionBoundary(new double[]{-1, -0.05, 0.08, 0.13, 1.5, 1.5});
		hypotheses[1] = new QuadraticDecisionBoundary(new double[]{-1, -0.05, 0.08, 0.13, 1.5, 15});
		hypotheses[2] = new QuadraticDecisionBoundary(new double[]{-1, -0.05, 0.08, 0.13, 15, 1.5});
		hypotheses[3] = new QuadraticDecisionBoundary(new double[]{-1, -1.5, 0.08, 0.13, 0.05, 0.05});
		hypotheses[4] = new QuadraticDecisionBoundary(new double[]{1, -0.05, 0.08, 1.5, 0.15, 0.15});
		return hypotheses;
	}
	
	public static void HW6_Q2(String data_file, String test_file) throws IOException {
		String[] tokens = null;
		ArrayList<String> lines = read_dataset(data_file);
		int N = lines.size();
		double[][] data = new double[N][3];
		int[] labels = new int[N];
		for(int i = 0; i < N; i++) {
			tokens = lines.get(i).split("\\s+");
			data[i][0] = 1;
			data[i][1] = Double.parseDouble(tokens[0]);
			data[i][2] = Double.parseDouble(tokens[1]);
			labels[i] = (int) Double.parseDouble(tokens[2]);
		}
		data = HW6Transformer.transform(data);
		Model model = get_model(data, labels);
		double in_error = calc_error(model, data, labels);
		//StdOut.println(data.length);
		lines = read_dataset(test_file);
		N = lines.size();
		data = new double[N][3];
		labels = new int[N];
		for(int i = 0; i < N; i++) {
			tokens = lines.get(i).split("\\s+");
			data[i][0] = 1;
			data[i][1] = Double.parseDouble(tokens[0]);
			data[i][2] = Double.parseDouble(tokens[1]);
			labels[i] = (int) Double.parseDouble(tokens[2]);
		}
		data = HW6Transformer.transform(data);
		//StdOut.println(data.length);
		double out_error = calc_error(model, data, labels);
		StdOut.println("In sample error = " + in_error);
		StdOut.println("Out of sample error = " + out_error);
	}
	
	public static void linear_target_test(int N, int test_examples_number, int number_of_experiments) {
		double out_error_rate = 0;
		double in_error_rate = 0;
		int curr = 0;
		//perform number of statistical experiments
		for(int n = 0; n < number_of_experiments; n++) {
			//print debug information
			if((curr++)%50 == 0) {
				StdOut.println(curr-1);
			}
			//train regression model
			Model target_function = new LinearDecisionBoundary();
			double[][] training_set = get_dataset(N);
			int[] labels = get_labels(training_set, target_function);
			Model regression_model = get_model(training_set, labels);
			in_error_rate += calc_error(regression_model, training_set, labels); 
			//compute error rate on new data set
			double[][] test_set = get_dataset(test_examples_number);
			int[] test_labels = get_labels(test_set, target_function);
			out_error_rate += calc_error(regression_model, test_set, test_labels); 
		}
		//print summary for a particular size of training set
		StdOut.println("For N = " + N + ":");
		StdOut.println("In sample Error rate: " + in_error_rate/number_of_experiments);
		StdOut.println("Out of sample Error rate: " + out_error_rate/number_of_experiments);
	}
	
	public static void nonlinear_target_test(int N, int test_examples_number, int number_of_experiments) {
		double out_error_rate = 0;
		double in_error_rate = 0;
		int curr = 0;
		//Question 9 in the HW2
		Model[] hypotheses = get_hypotheses();
		int hypo_num = hypotheses.length;
		double[] errors = new double[hypo_num];
		double[] weights = new double[6];
		for(int n = 0; n < number_of_experiments; n++) {
			if((curr++)%50 == 0) {
				StdOut.println(curr-1);
			}
			double[][] training_set = get_dataset(N);
			int[] labels = get_labels(training_set, new CircleDecisionBoundary(0.6));
			//add noise to the labels
			add_noise(labels, 0.1);
			training_set = Quadratic2DTransformer.transform(training_set);
			Model model = get_model(training_set, labels);
			double[] w = model.get_weights();	
			for(int j = 0; j < 6; j++) {
				weights[j] += w[j];
			}
			double[][] test_set = get_dataset(test_examples_number);
			test_set = Quadratic2DTransformer.transform(test_set);
			int[][] hypo_labels = new int[hypo_num][test_examples_number];
			for(int i = 0; i < hypo_num; i++) {
				hypo_labels[i] = get_labels(test_set, hypotheses[i]);
			}
			for(int i = 0; i < hypo_num; i++) {
				errors[i] += calc_error(model, test_set, hypo_labels[i]); 
			} 
		}
		curr = 0;
		//perform number of statistical experiments for the Q10 in HW2
		for(int n = 0; n < number_of_experiments; n++) {
			//print debug information
			if((curr++)%50 == 0) {
				StdOut.println(curr-1);
			}
			//train regression model
			double[][] training_set = get_dataset(N);
			int[] labels = get_labels(training_set, new CircleDecisionBoundary(0.6));
			//add noise to the labels
			add_noise(labels, 0.1);
			training_set = Quadratic2DTransformer.transform(training_set);
			Model w = get_model(training_set, labels);
			//calculate in sample error
			in_error_rate += calc_error(w, training_set, labels); 
			//compute error rate on new data set
			double[][] test_set = get_dataset(test_examples_number);
			int[] test_labels = get_labels(test_set, new CircleDecisionBoundary(0.6));
			add_noise(test_labels, 0.1);
			test_set = Quadratic2DTransformer.transform(test_set);
			out_error_rate += calc_error(w, test_set, test_labels);
		} 
		//print summary for a particular size of training set
		StdOut.println("\n*** Summary for N = " + N + " *** \n");
		StdOut.println(" - In-sample error rate: " + in_error_rate/number_of_experiments);
		StdOut.println(" - Out-of-sample error rate: " + out_error_rate/number_of_experiments);
		StdOut.println("\n*** Average agreement errors *** \n");
		for(int i = 0; i < hypo_num; i++) {
			StdOut.println(" - on hypotheses " + i + ": " + errors[i]/number_of_experiments);
		}
		for(int i = 0; i < 6; i++) {
			weights[i] = weights[i]/number_of_experiments;
		}
		for(int i = 0; i < 6; i++) {
			weights[i] = -weights[i]/weights[0];
		}
		StdOut.println("\n*** Average weights *** \n");
		for(int i = 0; i < 6; i++) {
			StdOut.println(" - weight " + i + " = " + weights[i]);
		}
	}
	
	public static void main(String[] args) throws IOException {
		/* int N = Integer.parseInt(args[0]);
		int test_examples_number = Integer.parseInt(args[1]);
		int number_of_experiments = Integer.parseInt(args[2]); */
		//linear_target_test(N, test_examples_number, number_of_experiments);
		//nonlinear_target_test(N, test_examples_number, number_of_experiments);
		HW6_Q2(args[0], args[1]);
	}
}