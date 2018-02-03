class LinearRegression {
	
	public static double[] train(double[][] data, int[] labels, double lambda) {
		SimpleMatrix X = new SimpleMatrix(data);
		SimpleMatrix X_prime = X.transpose();
		return ((((X_prime.multiply(X)).plus(lambda)).pinv()).multiply(X_prime)).multiply(labels);
	}
	
	public static double dot(double[] x, double[] y) {
		double dot = 0;
		for(int i = 0; i < x.length; i++) {
			dot += x[i]*y[i];
		}
		return dot;
	}
	
	public static int sign(double x) {
		if(x > 0) {
			return 1;
		}
		return -1;
	}
	
	public static int classify(double[] w, double[] x) {
		return sign(dot(w, x));
	}
	
	public static double calc_error(double[] w, double[][] data, int[] labels) {
		double error = 0;
		for(int i = 0; i < data.length; i++) {
			if(classify(w, data[i]) != labels[i]) {
				error++;
			}
		}
		return error/data.length;
	}
}