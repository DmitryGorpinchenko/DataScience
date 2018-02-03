abstract class Kernel {

	abstract public double compute(double[] x1, double[] x2);
	
	//computes the kernel matrix Q
	public double[][] transform(double[][] data, int[] labels) {
		int N = data.length;
		double[][] Q = new double[N][N];
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < N; j++) {
				Q[i][j] = labels[i]*labels[j]*compute(data[i], data[j]);
			}
		}
		return Q;
	}
}