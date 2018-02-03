class DecisionStump {
	
	public double theta = Double.NEGATIVE_INFINITY;
	public int decision_feature;
	public int s;
	public double E_in;
	double weighted_error;
	
	public DecisionStump(double[][] data, int[] labels, double[] weights) {
		int N = data.length;
		int d = data[0].length;
		double pos = 0, neg = 0;
		for(int i = 0; i < N; i++) {
			if(labels[i] == 1) {
				pos += weights[i];
			} else {
				neg += weights[i];
			}
		}
		if(pos >= neg) {
			weighted_error = neg;
			s = 1;
		} else {
			weighted_error = pos;
			s = -1;
		}
		for(int id = 0; id < d; id++) {
			sort(data, labels, weights, id);
			double left_pos = 0, left_neg = 0;
			for(int e = 0; e < N-1; e++) {
				double curr_thresh = data[e][id] + (data[e+1][id] - data[e][id])/2;
				if(labels[e] == 1) {
					left_pos += weights[e];
				} else {
					left_neg += weights[e]; 
				}
				double curr_right_error = left_pos + (neg - left_neg);
				double curr_left_error = left_neg + (pos - left_pos);
				double curr_error = Math.min(curr_right_error, curr_left_error);
				if(curr_error < weighted_error) {
					theta = curr_thresh;
					decision_feature = id;
					weighted_error = curr_error;
					s = (curr_right_error == curr_error) ? 1:-1;
				}
			}
		}
		E_in = calc_error(data, labels);
	}
	
	public int classify(double[] x) {
		return s*sign(x[decision_feature] - theta);
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
	
	public static void sort(double[][] data, int[] labels, double[] weights, int id) {
		int N = data.length;
		int d = data[0].length;
		for(int i = 0; i < N; i++) {
			for(int j = i; j > 0; j--) {
				if(data[j][id] < data[j-1][id]) {
					double[] temp = data[j];
					data[j] = data[j-1];
					data[j-1] = temp;
					int t = labels[j];
					labels[j] = labels[j-1];
					labels[j-1] = t;
					double tmp = weights[j];
					weights[j] = weights[j-1];
					weights[j-1] = tmp;
				} else {
					break;
				}
			}
		}
	}
	
	public int sign(double a) {
		if(a >= 0) {
			return 1;
		}
		return -1;
	}
}