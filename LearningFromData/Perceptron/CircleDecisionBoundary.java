class CircleDecisionBoundary extends Model {
	
	private double R;
	
	public CircleDecisionBoundary(double R) {
		this.R = R;
	}
	
	public int classify(double[] x) {
		return sign(x[1]*x[1] + x[2]*x[2] - R);
	}
}