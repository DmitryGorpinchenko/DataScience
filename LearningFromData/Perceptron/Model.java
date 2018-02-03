class Model {

	protected double[] weights;
	
	protected int sign(double x) {
		if(x < 0) {
			return -1;
		}	
		return 1;
	}
	
	protected double dot(double[] x1, double[] x2) {
		double dot = 0.0;
		for(int i = 0; i < x1.length; i++) {
			dot += x1[i]*x2[i];
		}
		return dot;
	}
	
	public void adjust_weights(double[] x, int a) {
		for(int i = 0; i < weights.length; i++) {
			weights[i] = weights[i] + a*x[i];
		}
	}
	
	public int classify(double[] x) {
		return sign(dot(weights, x));
	}
	
	public double[] get_weights() {
		return weights;
	}
}