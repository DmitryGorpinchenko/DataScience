class HW6Transformer {
	
	public static double[][] transform(double[][] data) {
		int N = data.length;
		double[][] newData = new double[N][8];
		for(int i = 0; i < N; i++) {
			newData[i][0] = data[i][0];
			newData[i][1] = data[i][1];
			newData[i][2] = data[i][2];
			newData[i][3] = data[i][1]*data[i][1];
			newData[i][4] = data[i][2]*data[i][2];
			newData[i][5] = data[i][1]*data[i][2];
			newData[i][6] = Math.abs(data[i][1] - data[i][2]);
			newData[i][7] = Math.abs(data[i][1] + data[i][2]);
		}
		return newData;
	}
}