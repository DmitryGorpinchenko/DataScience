class LogisticResgression {
	
	public static double[] get_model(double[][] dataset, int[] labels) {
		double rate = 0.01;
		double tol = 0.01;
		double[] w0 = new double[dataset[0].length];
		Function[] errors = new PartialErrorFunction[dataset.length];
		for(int i = 0; i < dataset.length; i++) {
			errors[i] = new PartialErrorFunction(dataset[i], labels[i]);
		}
		return GradientDescent.SGD(errors, w0, rate, tol);
	}
	
	public static double dot(double[] a, double[] b) {
		double dot = 0;
		for(int i = 0; i < a.length; i++) {
			dot += a[i]*b[i];
		}
		return dot;
	}
	
	public static double sigmoid(double s) {
		return 1.0/(1 + Math.exp(-s));
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
	
	public static int[] get_labels(double[][] dataset, double[] f) {
		int N = dataset.length;
		int[] labels = new int[N];
		for(int i = 0; i < N; i++) {
			double val = dataset[i][2] - (f[0]*dataset[i][1] + f[1]);
			if(val >= 0) {
				labels[i] = 1;
			} else {
				labels[i] = -1;
			}
		}
		return labels;
	}
	
	public static double[] get_linear_target() {
		double[] f = new double[2];
		double x1 = 2*Math.random() - 1;
		double x2 = 2*Math.random() - 1;
		double y1 = 2*Math.random() - 1;
		double y2 = 2*Math.random() - 1;
		f[0] = (y2 - y1)/(x2 - x1);
		f[1] = y2 - f[0]*x2;
		return f;
	}
	
	public static double calc_cross_entropy_error(double[] w, double[][] test_data, int[] test_labels) {
		double error = 0;
		for(int i = 0; i < test_data.length; i++) {
			error += new PartialErrorFunction(test_data[i], test_labels[i]).get_value(w);
		}
		return error/test_data.length;
	}
	
	public static void test_logistic(int N, int exper_num, int test_examples_num) {
		double out_error = 0.0;
		for(int i = 0; i < exper_num; i++) {
			if(i%10 == 0) StdOut.println(i);
			double[] f = get_linear_target();
			double[][] dataset = get_dataset(N);
			int[] labels = get_labels(dataset, f);
			double[] w = get_model(dataset, labels);
			double[][] test_data = get_dataset(test_examples_num);
			int[] test_labels = get_labels(test_data, f);
			out_error += calc_cross_entropy_error(w, test_data, test_labels);
		}
		StdOut.println("Average epochs number = " + (GradientDescent.epoch_num + 0.0)/exper_num);
		StdOut.println("E_out = " + out_error/exper_num);
	}
	
	public static void main(String[] args) {
		int N = Integer.parseInt(args[0]);
		int exper_num = Integer.parseInt(args[1]);
		int test_examples_num = Integer.parseInt(args[2]);
		test_logistic(N, exper_num, test_examples_num);
	}
}