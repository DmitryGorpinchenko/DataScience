class PolynomialKernel extends Kernel {
	
	private int Q;
	
	public PolynomialKernel(int Q) {
		this.Q = Q;
	}
	
	public double compute(double[] x1, double[] x2) {
		double temp = 1;
		for(int i = 0; i < x1.length; i++) {
			temp += x1[i]*x2[i];
		}
		double val = 1;
		for(int i = 0; i < Q; i++) {
			val *= temp;
		}
		return val;
	}
}