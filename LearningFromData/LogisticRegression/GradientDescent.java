class GradientDescent {

	public static int iteration_num = 0;
	public static double min_val = 0;
	public static int epoch_num = 0;
	
	public static void reset() {
		iteration_num = 0;
		min_val = 0;
		epoch_num = 0;
	}
	
	public static double[] sum(double[] a, double[] b) {
		int size = a.length;
		double[] sum = new double[size];
		for(int i = 0; i < size; i++) {
			sum[i] = a[i] + b[i];
		}
		return sum;
	}
	
	public static double[] scale(double[] a, double rate) {
		double[] a_scale = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			a_scale[i] = rate*a[i];
		}
		return a_scale;
	}
	
	public static double[] optimize_by_function_value(Function f, double[] w0, double rate, double tol) {
		double[] opt = w0;
		double[] grad = null;
		while(f.get_value(opt) >= tol) {
			grad = f.get_gradient(opt);
			opt = sum(opt, scale(grad, -rate));
			iteration_num++;
		}
		return opt;
	}
	
	public static double[] optimize_by_coordinate_descent(Function f, double[] w0, double rate, int iter_num) {
		int size = w0.length;
		double[] opt = w0;
		double grad = 0;
		for(int i = 0; i < iter_num; i++) {
			for(int k = 0; k < size; k++) {
				grad = f.get_gradient(opt, k);
				opt[k] = opt[k] - rate*grad;
			}
		}
		min_val = f.get_value(opt);
		return opt;
	}
	
	public static double L1_norm(double[] x) {
		double L1 = 0;
		for(int i = 0; i < x.length; i++) {
			L1 += Math.abs(x[i]);
		}
		return L1;
	}
	
	public static double L2_norm(double[] x) {
		double L2 = 0;
		for(int i = 0; i < x.length; i++) {
			L2 += x[i]*x[i];
		}
		return Math.sqrt(L2);
	}
	
	public static double[] SGD(Function[] f, double[] w0, double rate, double tol) {
		int size = f.length;
		double[] opt = w0;
		double[] old = null;
		double[] grad = null;
		while(true) {
			epoch_num++;
			old = opt;
			//permute a f functions
			for(int i = 0; i < size; i++) {
				int id = StdRandom.uniform(i, size);
				Function temp = f[id];
				f[id] = f[i];
				f[i] = temp;
			}
			for(int i = 0; i < size; i++) {
				grad = f[i].get_gradient(opt);
				opt = sum(opt, scale(grad, -rate));
			}
			if(L2_norm(sum(old, scale(opt, -1))) < tol) {
				break;
			}
		}
		return opt;
	}
	
	public static void main(String[] args) {
		/* double[] w = optimize_by_function_value(new TestSurface(), new double[] {1, 1}, 0.1, 1e-14);
		StdOut.println(iteration_num);
		StdOut.println("u = " + w[0] + ", v = " + w[1]); */
		double[] w = optimize_by_coordinate_descent(new TestSurface(), new double[] {1, 1}, 0.1, 15);
		StdOut.println(min_val);
	}
}