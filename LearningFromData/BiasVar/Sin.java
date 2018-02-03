class Sin implements TargetFunction {

	private double a;
	
	public Sin(double a) {
		this.a = a;
	}

	public double get_value(double x) {
		return Math.sin(a*x);
	}
}