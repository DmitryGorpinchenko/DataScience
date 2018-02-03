class DecisionTree {

	private Node root;
	private int features_num;
	public int node_count;
	private String alg_type;
	private int max_depth;
	
	private class Node {
		private int decision_feature;
		private double threshold;
		private int prediction;
		private Node left;
		private Node right;
	}
	
	public DecisionTree(String alg_type, int max_depth) {
		this.alg_type = alg_type;
		this.max_depth = max_depth;
	}
	
	public void train(double[][] data, int[] labels) {
		if("CART".equals(alg_type)) {
			CART(data, labels);
		}
	}
	
	public void CART(double[][] data, int[] labels) {
		features_num = data[0].length;
		root = learn_subtrees(data, labels, 0);
	}
	
	public Node learn_subtrees(double[][] data, int[] labels, int depth) {
		int N = data.length;
		Node x = new Node();
		int pos = 0, neg = 0;
		for(int i = 0; i < N; i++) {
			if(labels[i] == 1) {
				pos++;
			} else {
				neg++;
			}
		}
		if(depth == max_depth || all_labels_the_same(labels) || all_examples_the_same(data)) {
			x.prediction = pos > neg ? 1:-1;
			return x;
		}
		int decision_feature = -1;
		double thresh = Double.NEGATIVE_INFINITY; 
		double best_gini = 1 - (pos+0.0)*(pos+0.0)/(N*N) - (neg+0.0)*(neg+0.0)/(N*N);//Double.POSITIVE_INFINITY;
		int l_N = -1, r_N = -1;
		for(int id = 0; id < features_num; id++) {
			sort(data, labels, id);
			int left_pos = 0, left_neg = 0;
			for(int i = 0; i < N-1; i++) {
				if(labels[i] == 1) {
					left_pos++;
				} else {
					left_neg++;
				}
				double curr_gini = (i+1.0)/N*(1-(left_pos+0.0)*(left_pos+0.0)/((i+1)*(i+1)) - (left_neg+0.0)*(left_neg+0.0)/((i+1)*(i+1))) + 
								   (N-(i+1.0))/N*(1-(pos-left_pos+0.0)*(pos-left_pos+0.0)/((N-i-1)*(N-i-1)) - (neg-left_neg+0.0)*(neg-left_neg+0.0)/((N-i-1)*(N-i-1)));			   
				double curr_thresh = data[i][id] + (data[i+1][id] - data[i][id])/2;
				if(curr_gini < best_gini) {
					best_gini = curr_gini;
					thresh = curr_thresh;
					decision_feature = id;
					l_N = i+1;
					r_N = N-i-1;
				}				
			}
		}
		if(thresh == Double.NEGATIVE_INFINITY) {
			x.prediction = pos > neg ? 1:-1;
			return x;
		}
		sort(data, labels, decision_feature);
		node_count++;
		double[][] left_data = new double[l_N][];
		double[][] right_data = new double[r_N][];
		int[] left_labels = new int[l_N];
		int[] right_labels = new int[r_N];
		for(int i = 0; i < l_N; i++) {
			left_data[i] = data[i];
			left_labels[i] = labels[i];
		}
		for(int i = 0; i < r_N; i++) {
			right_data[i] = data[i+l_N];
			right_labels[i] = labels[i+l_N];
		}
		x.decision_feature = decision_feature;
		x.threshold = thresh; 
		x.left = learn_subtrees(left_data, left_labels, depth+1);
		x.right = learn_subtrees(right_data, right_labels, depth+1);
		return x;
	}
	
	public int classify(double[] x) {
		return classify(root, x);
	}
	
	public int classify(Node node, double[] x) {
		if(node.prediction != 0) {
			return node.prediction;
		}
		if(x[node.decision_feature] < node.threshold) {
			return classify(node.left, x);
		} else {
			return classify(node.right, x);
		}
	}
	
	public int get_node_count() {
		return node_count;
	}
	
	private boolean all_examples_the_same(double[][] data) {
		for(int i = 0; i < data.length-1; i++) {
			if(!is_equal(data[i], data[i+1])) {
				return false;
			}
		}
		return true;
	}
	
	private boolean all_labels_the_same(int[] labels) {
		for(int i = 0; i < labels.length-1; i++) {
			if(labels[i] != labels[i+1]) {
				return false;
			}
		}
		return true;
	}

	private boolean is_equal(double[] a, double[] b) {
		for(int i = 0; i < a.length; i++) {
			if(a[i] != b[i]) {
				return false;
			}
		}
		return true;
	}
	
	public static void sort(double[][] data, int[] labels, int id) {
		int N = data.length;
		for(int i = 0; i < N; i++) {
			for(int j = i; j > 0; j--) {
				if(data[j][id] < data[j-1][id]) {
					double[] temp = data[j];
					data[j] = data[j-1];
					data[j-1] = temp;
					int t = labels[j];
					labels[j] = labels[j-1];
					labels[j-1] = t;
				} else {
					break;
				}
			}
		}
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