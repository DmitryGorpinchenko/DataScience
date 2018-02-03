import Jama.Matrix;

class SimpleMatrix {
	private Matrix A;
	int m, n;
	
	public SimpleMatrix(double[][] a) {
		A = new Matrix(a);
		m = a.length;
		n = a[0].length;
	}
	
	public double get(int i, int j) {
		return A.get(i, j);
	}
	
	public static SimpleMatrix I(int n) {
		return new SimpleMatrix(Matrix.identity(n, n).getArray());
	}
	
	//adds a*I, where I is the identity matrix
	public SimpleMatrix plus(double a) {
		return new SimpleMatrix(Matrix.identity(m, n).times(a).plus(A).getArray());
	}
	
	public SimpleMatrix transpose() {
		return new SimpleMatrix(A.transpose().getArray());
	}
	
	public SimpleMatrix multiply(SimpleMatrix other) {
		return new SimpleMatrix(A.times(other.A).getArray());
	}
	
	public double[] multiply(double[] a) {
		int N = a.length;
		double[] w = new double[N];
		double[][] array = A.getArray();
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < N; j++) {
				w[i] += array[i][j]*a[j];
			}
		}
		return w;
	}
	
	public double[] multiply(int[] a) {
		int N = a.length;
		double[][] array = A.getArray();
		double[] w = new double[array.length];
		for(int i = 0; i < w.length; i++) {
			for(int j = 0; j < N; j++) {
				w[i] += array[i][j]*a[j];
			}
		}
		return w;
	}
	
	public SimpleMatrix pinv() {
		Matrix invA = A.inverse();
		return new SimpleMatrix(invA.getArray());
	}
	
	public String toString() {
		double[][] a = A.getArray();
		int rows = a.length, columns = a[0].length;
		String s = "";
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < columns; j++) {
				s += ("A(" + i +  ", " + j + ") = " + a[i][j] + "\n");
			}
			s += "\n";
		}
		return s;
	}
	
	public static void main(String[] args) {
		int N = Integer.parseInt(args[0]), M = Integer.parseInt(args[1]);
		double[][] a = new double[N][M];
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < M; j++) {
				a[i][j] = Math.random();
			}
		}
		SimpleMatrix A = new SimpleMatrix(a);
		/*StdOut.println(A.transpose().multiply(A).pinv().multiply(A.transpose()));
		StdOut.println("******* PINV ******"); */
		//StdOut.println(A.pinv()); 
		A.pinv();
		
	}
}