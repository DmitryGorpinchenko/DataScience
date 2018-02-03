class RBFTest {
	
	public static void set_random_data(double[][] data, int[] labels, int N) {
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < 2; j++) {
				data[i][j] = 2*Math.random() - 1;
			}
			labels[i] = sign(data[i][1] - data[i][0] + 0.25*Math.sin(Math.PI*data[i][0]));
		}
	}
	
	public static int sign(double a) {
		if(a > 0) {
			return 1;
		}
		return -1;
	}	
	
	public static void test1(int N, int T) {
		int non_separable = 0;
		for(int t = 0; t < T; t++) {
			if(t%100 == 0) {
				StdOut.println(t);
			}
			double[][] data = new double[N][2];
			int[] labels = new int[N];
			set_random_data(data, labels, N);
			SVM svm = new SVM(new RBFKernel(1.5), Double.POSITIVE_INFINITY);
			svm.train(data, labels);
			double err = svm.calc_error(data, labels);
			if(err != 0) {
				non_separable++;
			}
		}
		StdOut.println("Non separable by RBF Kernel in " + (non_separable + 0.0)/T + " cases");
	}
	
	public static void test2(int N, int test_size, int T, int K) {
		int svm_better = 0;
		double svm_err = Double.POSITIVE_INFINITY, rbf_err = Double.POSITIVE_INFINITY;
		for(int t = 0; t < T; t++) {
			double[][] train_data = new double[N][2];
			int[] train_labels = new int[N];
			set_random_data(train_data, train_labels, N);
			double[][] test_data = new double[test_size][2];
			int[] test_labels = new int[test_size];
			set_random_data(test_data, test_labels, test_size);
			RBFModel rbf = new RBFModel(1.5, K);
			if(rbf.train(train_data, train_labels)) {
				rbf_err = rbf.calc_error(test_data, test_labels);
				SVM svm = new SVM(new RBFKernel(1.5), Double.POSITIVE_INFINITY);
				svm.train(train_data, train_labels);
				svm_err = svm.calc_error(test_data, test_labels);
				if(svm_err < rbf_err) {
					svm_better++;
				}
			}
		}
		StdOut.println("SVM better than RBF in " + (svm_better+0.0)/T + " fraction of runs");
	}
	
	public static void test3(int T) {
		int zero_E_in = 0;
		for(int t = 0; t < T; t++) {
			double[][] train_data = new double[100][2];
			int[] train_labels = new int[100];
			set_random_data(train_data, train_labels, 100);
			RBFModel rbf = new RBFModel(1.5, 9);
			if(rbf.train(train_data, train_labels)) {
				double rbf_err = rbf.calc_error(train_data, train_labels);
				if(rbf_err == 0) {
					zero_E_in++;
				}
			}
		}
		StdOut.println("\nE_in = 0 in " + (zero_E_in+0.0)/T + " fraction of runs");
	}
	
	public static void test4(int test_size, int T) {
		int E_in_down_E_out_up = 0, E_in_up_E_out_down = 0, both_down = 0, both_up = 0;
		double E_in_9, E_in_12, E_out_9, E_out_12;
		int accepted_runs = 0;
		for(int t = 0; t < T; t++) {
			double[][] train_data = new double[100][2];
			int[] train_labels = new int[100];
			set_random_data(train_data, train_labels, 100);
			double[][] test_data = new double[test_size][2];
			int[] test_labels = new int[test_size];
			set_random_data(test_data, test_labels, test_size);
			RBFModel rbf9 = new RBFModel(1.5, 9);
			if(!rbf9.train(train_data, train_labels)) {
				continue;
			}
			RBFModel rbf12 = new RBFModel(1.5, 12);
			if(!rbf12.train(train_data, train_labels)) {
				continue;
			}
			accepted_runs++;
			E_in_9  = rbf9.calc_error(train_data, train_labels);
			E_in_12 = rbf12.calc_error(train_data, train_labels);
			E_out_9 = rbf9.calc_error(test_data, test_labels);
			E_out_12 = rbf12.calc_error(test_data, test_labels);
			if((E_in_9 < E_in_12) && (E_out_9 > E_out_12)) {
				E_in_up_E_out_down++;
			} else if((E_in_9 > E_in_12) && (E_out_9 < E_out_12)) {
				E_in_down_E_out_up++;
			} else if((E_in_9 < E_in_12) && (E_out_9 < E_out_12)) {
				both_up++;
			} else if((E_in_9 > E_in_12) && (E_out_9 > E_out_12)) {
				both_down++;
			}
		}
		StdOut.println("\n   *** When K goes from 9 to 12: ***\n");
		StdOut.println(String.format(java.util.Locale.UK, "E_in goes down but E_out goes up in %.3f fraction of runs",(E_in_down_E_out_up+0.0)/accepted_runs));
		StdOut.println(String.format(java.util.Locale.UK, "E_in goes up but E_out goes down in %.3f fraction of runs", (E_in_up_E_out_down+0.0)/accepted_runs));
		StdOut.println(String.format(java.util.Locale.UK, "Both E_in and E_out go up in %.3f fraction of runs", (both_up+0.0)/accepted_runs));
		StdOut.println(String.format(java.util.Locale.UK, "Both E_in and E_out go down in %.3f fraction of runs", (both_down+0.0)/accepted_runs));
	}
	
	public static void test5(int test_size, int T) {	
		int E_in_down_E_out_up = 0, E_in_up_E_out_down = 0, both_down = 0, both_up = 0;
		double E_in_15, E_in_20, E_out_15, E_out_20;
		int accepted_runs = 0;
		for(int t = 0; t < T; t++) {
			double[][] train_data = new double[100][2];
			int[] train_labels = new int[100];
			set_random_data(train_data, train_labels, 100);
			double[][] test_data = new double[test_size][2];
			int[] test_labels = new int[test_size];
			set_random_data(test_data, test_labels, test_size);
			RBFModel rbf15 = new RBFModel(1.5, 9);
			if(!rbf15.train(train_data, train_labels)) {
				continue;
			}
			RBFModel rbf20 = new RBFModel(2, 9);
			if(!rbf20.train(train_data, train_labels)) {
				continue;
			}
			accepted_runs++;
			E_in_15  = rbf15.calc_error(train_data, train_labels);
			E_in_20 = rbf20.calc_error(train_data, train_labels);
			E_out_15 = rbf15.calc_error(test_data, test_labels);
			E_out_20 = rbf20.calc_error(test_data, test_labels);
			if((E_in_15 < E_in_20) && (E_out_15 > E_out_20)) {
				E_in_up_E_out_down++;
			} else if((E_in_15 > E_in_20) && (E_out_15 < E_out_20)) {
				E_in_down_E_out_up++;
			} else if((E_in_15 < E_in_20) && (E_out_15 < E_out_20)) {
				both_up++;
			} else if((E_in_15 > E_in_20) && (E_out_15 > E_out_20)) {
				both_down++;
			} 
		}
		StdOut.println("\n   *** When Gamma goes from 1.5 to 2.0: ***\n");
		StdOut.println(String.format(java.util.Locale.UK, "E_in goes down but E_out goes up in %.3f fraction of runs",(E_in_down_E_out_up+0.0)/accepted_runs));
		StdOut.println(String.format(java.util.Locale.UK, "E_in goes up but E_out goes down in %.3f fraction of runs", (E_in_up_E_out_down+0.0)/accepted_runs));
		StdOut.println(String.format(java.util.Locale.UK, "Both E_in and E_out go up in %.3f fraction of runs", (both_up+0.0)/accepted_runs));
		StdOut.println(String.format(java.util.Locale.UK, "Both E_in and E_out go down in %.3f fraction of runs", (both_down+0.0)/accepted_runs));
	}
	
	public static void main(String[] args) {
		Stopwatch sw = new Stopwatch();
		//test1(Integer.parseInt(args[0]), Integer.parseInt(args[1]));
		//test2(Integer.parseInt(args[0]), Integer.parseInt(args[1]), Integer.parseInt(args[2]), Integer.parseInt(args[3]));
		//test3(Integer.parseInt(args[0]));
		test4(Integer.parseInt(args[0]), Integer.parseInt(args[1]));
		//test5(Integer.parseInt(args[0]), Integer.parseInt(args[1]));
		StdOut.println("\nTiming results: " + sw.elapsedTime());
	}
}