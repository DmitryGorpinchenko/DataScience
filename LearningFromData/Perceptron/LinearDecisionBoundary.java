class LinearDecisionBoundary extends Model {

	public LinearDecisionBoundary() {
		double x1 = 2*Math.random() - 1;
		double y1 = 2*Math.random() - 1;
		double x2 = 2*Math.random() - 1;
		double y2 = 2*Math.random() - 1;
		double a = (y2 - y1)/(x2 - x1);
		double b = y2 - a*x2;
		weights = new double[3];
		weights[0] = -b;
		weights[1] = -a;
		weights[2] = 1;
	}
	
	public LinearDecisionBoundary(int d) {
		weights = new double[d];
	}
	
	public LinearDecisionBoundary(double[] weights) {
		this.weights = weights;
	}
}