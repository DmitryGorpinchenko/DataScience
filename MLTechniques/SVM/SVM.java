import java.util.HashMap;

class SVM {
		
	public double C;
	public Kernel kernel;
	public double[] alpha;
	public HashMap<Integer, double[]> sv = new HashMap<>();
	public double b; 
	public int N;
	
	public SVM(Kernel kernel, double C) {
		this.C = C;
		this.kernel = kernel;
	}
	
	public int classify(double[] x) {
		double val  = f(x);
		val = val + b;
		if(val > 0) {
			return 1;
		} 	
		return -1;
	}
	
	//suitable for the new data points x for which there is no known kernel matrix in advance
	public double f(double[] x) {
		double val = 0;
		for(int i : sv.keySet()) {
			double[] support_vector = sv.get(i);
			val += alpha[i]*kernel.compute(support_vector, x);
		}
		return val;
	}
	
	//suitable for random experimentation through index selection by shuffling
	//by selecting the elements from the Q matrix have computed in advance only once via id array
	public void train(double[][] Q, double[][] data, int[] labels, int[] id) {
		N = id.length;                                           
		alpha = SMO.smo(Q, labels, C, id);
		//set support vectors
		for(int i = 0; i < N; i++) {
			if(alpha[i] > 0) {
				sv.put(i, data[id[i]]);
				alpha[i] = alpha[i]*labels[id[i]]; //in order to avoid memorizing of data labels
			}
		}
		set_bias(data, labels, id);
	}
	
	public void set_bias(double[][] data, int[] labels, int[] id) {
		//set up the threshold b
		for(int i : sv.keySet()) {
			if(Math.abs(alpha[i]) > 0 && Math.abs(alpha[i]) < C) {
				b = labels[id[i]] - f(data[id[i]]);
				return;
			}		
		}
		double begin = Double.NEGATIVE_INFINITY, end = Double.POSITIVE_INFINITY;
		for(int i = 0; i < N; i++) {
			double temp = labels[id[i]] - f(data[id[i]]);
			if(Math.abs(alpha[i]) == C) {
				if(labels[id[i]] == 1) {
					if(end > temp) {
						end = temp;
					}
				} else {
					if(begin < temp) {
						begin = temp;
					}
				}
			} else if(Math.abs(alpha[i]) == 0) {
				if(labels[id[i]] == 1) {
					if(begin < temp) {
						begin = temp;
					}
				} else {
					if(end > temp) {
						end = temp;
					}
				}
			}
		}
		b = begin + (end - begin)/2;
	}
	
	public void train(double[][] data, int[] labels) {
		N = data.length;                                           
		double[][] Q = kernel.transform(data, labels); //get kernel matrix
		alpha = SMO_WSS3.smo_wss3(Q, labels, C);
		//support vectors
		for(int i = 0; i < N; i++) {
			if(alpha[i] > 0) {
				sv.put(i, data[i]);
				alpha[i] = alpha[i]*labels[i]; //in order to avoid memorizing of data labels
			}
		}
		//set up the threshold b
		for(int i : sv.keySet()) {
			if(Math.abs(alpha[i]) > 0 && Math.abs(alpha[i]) < C) {
				b = labels[i] - f(data[i]);
				return;
			}		
		}
		double begin = Double.NEGATIVE_INFINITY, end = Double.POSITIVE_INFINITY;
		for(int i = 0; i < N; i++) {
			double temp = labels[i] - f(data[i]);
			if(Math.abs(alpha[i]) == C) {
				if(labels[i] == 1) {
					if(end > temp) {
						end = temp;
					}
				} else {
					if(begin < temp) {
						begin = temp;
					}
				}
			} else if(Math.abs(alpha[i]) == 0) {
				if(labels[i] == 1) {
					if(begin < temp) {
						begin = temp;
					}
				} else {
					if(end > temp) {
						end = temp;
					}
				}
			}
		}
		b = begin + (end - begin)/2;
	} 
	
	public double calc_error(double[][] test_data, int[] test_labels) {
		int incorrect = 0;
		for(int i = 0; i < test_data.length; i++) {
			if(classify(test_data[i]) != test_labels[i]) {
				incorrect++;
			}
		}
		return (incorrect+0.0)/test_data.length;
	}
	
	public double abs_w() {
		double w = 0;
		for(int id1 : sv.keySet()) {
			for(int id2 : sv.keySet()) {
				double[] x1 = sv.get(id1);
				double[] x2 = sv.get(id2);
				w += alpha[id1]*alpha[id2]*kernel.compute(x1, x2);
			}
		}
		return Math.sqrt(w);
	}
	
	public double obj() {
		double sum_alpha = 0;
		for(int id : sv.keySet()) {
			sum_alpha += Math.abs(alpha[id]);
		}
		return abs_w()*abs_w()/2 - sum_alpha;
	}
}