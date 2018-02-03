class PartialErrorFunction implements Function {

	private double[] x;
	private int y;
	
	public PartialErrorFunction(double[] x, int y) {
		this.x = x;
		this.y = y;
	}
	
	private double dot(double[] a, double[] b) {
		double dot = 0;
		for(int i = 0; i < a.length; i++) {
			dot += a[i]*b[i];
		}
		return dot;
	}
	
	public double get_value(double[] w) {
		return Math.log(1 + Math.exp(-y*dot(x, w)));
	}
	
	public double[] get_gradient(double[] w) {
		int size = w.length;
		double dot = dot(w, x);
		double[] grad = new double[size];
		for(int i = 0; i < size; i++) {
			grad[i] = -y*x[i]/(1+Math.exp(y*dot));
		}
		return grad;
	}
	
	public double get_gradient(double[] w, int i){
		return -y*x[i]/(1+Math.exp(y*dot(x, w)));
	}
}