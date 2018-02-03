class RandomForest {
	
	private DecisionTree[] forest;
	private int T;
	private int max_depth;
	
	public RandomForest(int T, int max_depth) {
		this.T = T;
		this.max_depth = max_depth;
		forest = new DecisionTree[T];
	}
	
	public void train(double[][] data, int[] labels) {
		for(int i = 0; i < T; i++) {
			int[] ids = get_bootstrap_ids(data.length);
			double[][] train_data = new double[data.length][];
			int[] train_labels = new int[data.length];
			for(int j = 0; j < data.length; j++) {
				train_data[j] = data[ids[j]];
				train_labels[j] = labels[ids[j]];
			}
			forest[i] = new DecisionTree("CART", max_depth);
			forest[i].train(train_data, train_labels);
		}
	}
	
	public int[] get_bootstrap_ids(int N) {
		int[] ids = new int[N];
		for(int i = 0; i < N; i++) {
			ids[i] = StdRandom.uniform(0, N);
		}
		return ids;
	}
	
	public int classify(double[] x) {
		int s = 0;
		for(int i = 0; i < T; i++) {
			s += forest[i].classify(x);
		}
		return sign(s);
	}
	
	public static int sign(double a) {
		if(a >= 0) {
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
	
	public double calc_error(double[][] data, int[] labels, int tree_id) {
		return forest[tree_id].calc_error(data, labels);
	}
}