import java.util.HashMap;
import java.util.ArrayList;

class RBFModel {
	
	public int K;
	public int d;
	public double[][] mu;  
	public double[] w;
	public double b;
	public Kernel rbf;
	
	public RBFModel(double gamma, int K) {
		this.K = K;
		rbf = new RBFKernel(gamma);
	}
	
	//returns a flag whether or not clustering is valid (no empty clusters) (if 'false' learning is not happens!!!)
	public boolean train(double[][] data, int[] labels) {
		d = data[0].length;
		mu = new double[K][d];
		w = new double[K];
		if(cluster(data)) {
			find_weights(data, labels);
			return true;
		}
		return false;
	}
	
	public void find_weights(double[][] data, int[] labels) {
		int N = data.length;
		double[][] Q = new double[N][K+1];
		for(int i = 0; i < N; i++) {
			Q[i][0] = 1;
			for(int j = 1; j <= K; j++) {
				Q[i][j] = rbf.compute(data[i], mu[j-1]);
			}
		}
		SimpleMatrix m = new SimpleMatrix(Q);
		double[] weights = m.pinv().multiply(labels);
		b = weights[0];
		for(int i = 0; i < K; i++) {
			w[i] = weights[i+1];
		}
	}
	
	//assumes that domain is cubic [-1, 1] x [-1, 1] x ... x [-1, 1]
	public boolean cluster(double[][] data) {
		int N = data.length;
		for(int i = 0; i < K; i++) {
			for(int j = 0; j < d; j++) {
				mu[i][j] = 2*Math.random() - 1;
			}
		}
		HashMap<Integer, ArrayList<double[]>> clusters = null;
		while(true) {
			clusters = new HashMap<>();
			for(int i = 0; i < K; i++) {
				clusters.put(i, new ArrayList<double[]>());
			}
			//store centers before update
			double[][] old_mu = new double[K][d];
			for(int i = 0; i < K; i++) {
				for(int j = 0; j < d; j++) {
					old_mu[i][j] = mu[i][j];
				}
			}
			//assign points to centers
			for(int i = 0; i < N; i++) {
				int cluster_id = -1;
				double min_dist = Double.POSITIVE_INFINITY;
				for(int k = 0; k < K; k++) {
					double dist = dist(data[i], mu[k]);
					if(dist < min_dist) {
						min_dist = dist;
						cluster_id = k;
					}
 				}	
				clusters.get(cluster_id).add(data[i]);
			}
			//update centers
			for(int i = 0; i < K; i++) {
				if(clusters.get(i).size() == 0) {
					return false;
				}
				mu[i] = calc_mu(clusters.get(i));
			}
			if(!is_mu_changed(old_mu)) {
				return true;
			}
		}
	}
	
	public double[] calc_mu(ArrayList<double[]> cluster) {
		int size = cluster.size();
		double[] mu = new double[d];
		for(double[] point : cluster) {
			for(int i = 0; i < d; i++) {
				mu[i] += point[i]; 
			}
		}
		for(int i = 0; i < d; i++) {
			mu[i] /= size;
		}
		return mu;
	}
	
	public boolean is_mu_changed(double[][] old_mu) {
		for(int i = 0; i < K; i++) {
			for(int j = 0; j < d; j++) {
				if(mu[i][j] != old_mu[i][j]) {
					return true;
				}
			}
		}
		return false;
	}
	
	public int classify(double[] data) {
		int val = 0;
		for(int i = 0; i < K; i++) {
			val += w[i]*rbf.compute(data, mu[i]);
		}
		val += b;
		if(val > 0) {
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

	public double dist(double[] x, double[] mu) {
		double dist = 0;
		for(int i = 0; i < d; i++) {
			dist += (x[i] - mu[i])*(x[i] - mu[i]);
		}
		return dist;
	} 	
}