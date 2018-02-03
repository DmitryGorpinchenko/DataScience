class SMO_WSS3 {
	
	public static double eps = 1e-3;
	public static double tau = 1e-12;
	
	public static double[] smo_wss3(double[][] Q, int[] y, double C) {
		int len = y.length;
		double[] A = new double[len];
		double[] G = new double[len];
		for(int i = 0; i < len; i++) {
			G[i] = -1;
		}
		while(true) {
			int[] WS = selectB(Q, y, A, G, C);
			int i = WS[0], j = WS[1];
			if(j == -1) { 
				break;
			}
			
			double a = Q[i][i] + Q[j][j] - 2*y[i]*y[j]*Q[i][j];
			if(a <= 0) {
				a = tau;
			}
			double b = -y[i]*G[i]+y[j]*G[j];
			
			double oldAi = A[i], oldAj = A[j];
			A[i] += y[i]*b/a;
			A[j] -= y[j]*b/a;
			
			double sum = y[i]*oldAi+y[j]*oldAj;
			if(A[i] > C) {
				A[i] = C;
			} else if(A[i] < 0) {
				A[i] = 0;
			}
			A[j] = y[j]*(sum - y[i]*A[i]);
			if(A[j] > C) {
				A[j] = C;
			} else if(A[j] < 0) {
				A[j] = 0;
			}
			A[i] = y[i]*(sum-y[j]*A[j]);
			// update gradient
			double deltaAi = A[i] - oldAi, deltaAj = A[j] - oldAj;
			for(int t = 0; t < len; t++) {
				G[t] += Q[t][i]*deltaAi+Q[t][j]*deltaAj;
			}
		}
		return A;
	}
	
	public static int[] selectB(double[][] Q, int[] y, double[] A, double[] G, double C) {
		int len = y.length;
		// select i
		int i = -1;
		double G_max = Double.NEGATIVE_INFINITY;
		double G_min = Double.POSITIVE_INFINITY;
		for(int t = 0; t < len; t++) {
			if((y[t] == 1 && A[t] < C) || (y[t] == -1 && A[t] > 0)) {
				if(-y[t]*G[t] >= G_max) {
					i = t;
					G_max = -y[t]*G[t];
				}
			}
		}
		// select j
		int j = -1;
		double obj_min = Double.POSITIVE_INFINITY;
		for(int t = 0; t < len; t++) {
			if((y[t] == 1 && A[t] > 0) || (y[t] == -1 && A[t] < C)) {
				double b = G_max + y[t]*G[t];
				if(-y[t]*G[t] <= G_min) {
					G_min = -y[t]*G[t];
				}
				if(b > 0) {
					double a = Q[i][i]+Q[t][t]-2*y[i]*y[t]*Q[i][t];
					if (a <= 0) {
						a = tau;
					}
					if (-(b*b)/a <= obj_min) {
						j = t;
						obj_min = -(b*b)/a;
					}
				}
			}
		}
		if (G_max-G_min < eps) {
			return  new int[] {-1, -1};
		}
		return new int[] {i, j};
	}
}