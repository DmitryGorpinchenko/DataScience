import java.util.ArrayList;

class Perceptron {

	private Model model;
	private int N;
	private int d;
	public int iterations_to_converge;
	
	public Perceptron(double[][] train_data, int[] labels, boolean is_regression) {
		N = train_data.length;
		d = train_data[0].length;
		if(is_regression) {
			model = LinearRegression.get_model(train_data, labels);
		} else {
			model = new LinearDecisionBoundary(d);
		}
		while(true) {
			ArrayList<Integer> missclassified_ids = new ArrayList<>();
			for(int i = 0; i < N; i++) {
				if(model.classify(train_data[i]) != labels[i]) {
					missclassified_ids.add(i);
				}
			}
			if(missclassified_ids.isEmpty()) {
				break;
			}
			int missclassified_id = missclassified_ids.get(StdRandom.uniform(0, missclassified_ids.size()));
			((LinearDecisionBoundary) model).adjust_weights(train_data[missclassified_id], labels[missclassified_id]);
			iterations_to_converge++;
		}
	}
	
	public int predict(double[] x) {
		return model.classify(x);
	}
	
	public double calc_error(double[][] dataset, int[] labels) {
		int N = dataset.length;
		int incorrect = 0;
		for(int i = 0; i < N; i++) {
			if(predict(dataset[i]) != labels[i]) {
				incorrect++;
			}	
		} 
		return (incorrect + 0.0)/N; 
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
	
	public static void test_perceptron(int N, int test_examples_number, int number_of_experiments, boolean is_regression) {
		int average_iterations_to_converge = 0;
		double error_rate = 0;
		int curr = 0;
		//perform number of statistical experiments
		for(int n = 0; n < number_of_experiments; n++) {
			//print debug information
			if((curr++)%50 == 0) {
				StdOut.println(curr-1);
			}
			//train perceptron model
			Model target_function = new LinearDecisionBoundary();
			double[][] training_set = get_dataset(N);
			int[] labels = get_labels(training_set, target_function);
			Perceptron perc = new Perceptron(training_set, labels, is_regression);
			average_iterations_to_converge += perc.iterations_to_converge;
			//compute error rate on new data set
			double[][] test_set = get_dataset(test_examples_number);
			int[] test_labels = get_labels(test_set, target_function);
			error_rate += perc.calc_error(test_set, test_labels); 
		}
		//print summary for a particular size of training set
		StdOut.println("For N = " + N + ":");
		StdOut.println("Average iterations to converge: " + (average_iterations_to_converge+0.0)/number_of_experiments);
		StdOut.println("Error rate: " + error_rate/number_of_experiments);
	}
	
	public static void main(String[] args) {
		int N = Integer.parseInt(args[0]);
		int test_examples_number = Integer.parseInt(args[1]);
		int number_of_experiments = Integer.parseInt(args[2]);
		boolean is_regression = true;
		test_perceptron(N, test_examples_number, number_of_experiments, is_regression);
	}
}