class LinearKernel extends Kernel {
	
	public double compute(double[] x1, double[] x2) {
		double val = 0;
		for(int i = 0; i < x1.length; i++) {
			val += x1[i]*x2[i];
		}
		return val;
	}
}