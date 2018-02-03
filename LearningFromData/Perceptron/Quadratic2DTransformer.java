class Quadratic2DTransformer {
	
	public static double[][] transform(double[][] data) {
		int N = data.length;
		double[][] newData = new double[N][6];
		for(int i = 0; i < N; i++) {
			newData[i][0] = data[i][0];
			newData[i][1] = data[i][1];
			newData[i][2] = data[i][2];
			newData[i][3] = data[i][1]*data[i][2];
			newData[i][4] = data[i][1]*data[i][1];
			newData[i][5] = data[i][2]*data[i][2];
		}
		return newData;
	}
	
	public static double[] transform(double[] data) {
		int N = data.length;
		double[] newData = new double[6];
		newData[0] = data[0];
		newData[1] = data[1];
		newData[2] = data[2];
		newData[3] = data[1]*data[2];
		newData[4] = data[1]*data[1];
		newData[5] = data[2]*data[2];
		return newData;
	}
}