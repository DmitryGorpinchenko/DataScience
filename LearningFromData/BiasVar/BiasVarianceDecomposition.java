class BiasVarianceDecomposition {
	
	public static double learn(double[][] train_data) {
		double num = 0.0;
		double denom = 0.0;
		for(int i = 0; i < train_data.length; i++) {
			num += train_data[i][0]*train_data[i][1];
			denom += train_data[i][0]*train_data[i][0];
		}
		return num/denom;
	}
	
	public static double[][] generate_data(int N, TargetFunction f) {
		double[][] data = new double[N][2];
		for(int i = 0; i < N; i++) {
			data[i][0] = 2*Math.random() - 1;
			data[i][1] = f.get_value(data[i][0]);
		}
		return data;
	}
	
	public static double[] generate_hypotheses_set(int N, int size, TargetFunction f) {
		double[] a = new double[size];
		for(int i = 0; i < size; i++) {
			a[i] = learn(generate_data(N, f));
		}
		return a;
	}
	
	public static double calc_average_hypothesis(int N, double[] a) {
		double sum = 0.0;
		for(int i = 0; i < a.length; i++) {
			sum += a[i];
		}
		return sum/a.length;
	}
	
	public static double calc_bias(double a_bar, TargetFunction f, int points_num) {
		double bias = 0.0;
		for(int i = 0; i < points_num; i++) {
			double x = 2*Math.random() - 1;
			bias += (a_bar*x - f.get_value(x))*(a_bar*x - f.get_value(x));
		}
		return bias/points_num;
	}
	
	public static double calc_var(double a_bar, double[] hypotheses, int points_num) {
		double var = 0.0;
		for(int i = 0; i < hypotheses.length; i++) {
			var += (hypotheses[i] - a_bar)*(hypotheses[i] - a_bar);
		}
		var = var/hypotheses.length;
		double sum = 0;
		for(int i = 0; i < points_num; i++) {
			double x = 2*Math.random() - 1;
			sum += var*x*x;
		}
		return sum/points_num;
	}

	public static void bias_variance_decomposition(int N, int dataset_num, int points_num) {
		double[] a = generate_hypotheses_set(N, dataset_num, new Sin(Math.PI));
		double a_bar = calc_average_hypothesis(N, a);
		double bias = calc_bias(a_bar, new Sin(Math.PI), points_num);
		double var = calc_var(a_bar, a, points_num);
		StdOut.println("Average hypothesis: " + "g_bar(x) = " + a_bar + "x");
		StdOut.println("Bias = " + bias);
		StdOut.println("Var = " + var);
	}
	
	public static void main(String[] args) {
		int N = Integer.parseInt(args[0]);
		int dataset_num = Integer.parseInt(args[1]);
		int points_num = Integer.parseInt(args[2]);
		bias_variance_decomposition(N, dataset_num, points_num);
	}
}