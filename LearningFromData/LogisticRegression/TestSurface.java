class TestSurface implements Function {

	public double get_value(double[] w) {
		double u = w[0];
		double v = w[1];
		double a = (u*Math.exp(v) - 2*v*Math.exp(-u));
		return a*a;
	}
	
	public double[] get_gradient(double[] w) {
		double u = w[0];
		double v = w[1];
		double[] grad = new double[2];
		grad[0] = 2*(Math.exp(v) + 2*v*Math.exp(-u))*(u*Math.exp(v) - 2*v*Math.exp(-u));
		grad[1] = 2*(u*Math.exp(v) - 2*Math.exp(-u))*(u*Math.exp(v) - 2*v*Math.exp(-u));
		return grad;
	}
	
	public double get_gradient(double[] w, int i) {
		double u = w[0];
		double v = w[1];
		double grad = 0;
		if(i == 0) {
			grad = 2*(Math.exp(v) + 2*v*Math.exp(-u))*(u*Math.exp(v) - 2*v*Math.exp(-u));
		} else if(i == 1) {
			grad = 2*(u*Math.exp(v) - 2*Math.exp(-u))*(u*Math.exp(v) - 2*v*Math.exp(-u));
		}
		return grad;
	}
}