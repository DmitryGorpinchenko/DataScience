interface Function {
	
	public double get_value(double[] w);
	
	public double[] get_gradient(double[] w);
	
	public double get_gradient(double[] w, int id);
}