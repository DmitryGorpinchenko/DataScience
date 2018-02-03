class AdaBoostStump {
	
	public int T;
	public DecisionStump[] stumps;
	public double[] alphas;
	
	//for information about iterations
	public double[] weight_sums;
	
	public AdaBoostStump(int T) {
		this.T = T;
		stumps = new DecisionStump[T];
		alphas = new double[T];
		weight_sums = new double[T];
	}
	
	public void train(double[][] data, int[] labels) {
		int N = data.length;
		double[] weights = new double[N];
		for(int i = 0; i < N; i++) {
			weights[i] = 1.0/N;
		}		
		double weight_sum = 1;
		for(int t = 0; t < T; t++) {
			stumps[t] = new DecisionStump(data, labels, weights);
			weight_sums[t] = weight_sum;
			double epsilon = stumps[t].weighted_error/weight_sum;
			double scale = Math.sqrt((1-epsilon)/epsilon);
			weight_sum = 0;
			for(int i = 0; i < N; i++) {
				if(stumps[t].classify(data[i]) != labels[i]) {
					weights[i] *= scale;
				} else {
					weights[i] /= scale;
				}
				weight_sum += weights[i];
			}
			alphas[t] = Math.log(scale);
		}
	}
	
	public int classify(double[] x) {
		double y = 0;
		for(int i = 0; i < T; i++) {
			y += alphas[i]*stumps[i].classify(x);
		}
		if(y >= 0) {
			return 1;
		}
		return -1;
	}
	
	public double calc_error(double[][] data, int[] labels) {
		int incorrect = 0;
		for(int i = 0; i < data.length; i++) {
			if(classify(data[i]) != labels[i]) {
				incorrect++;
			}
		}
		return (incorrect+0.0)/data.length;
	}
}