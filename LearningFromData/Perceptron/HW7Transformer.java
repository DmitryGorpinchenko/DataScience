class HW7Transformer {
	
	public static double[][] transform(double[][] data, int k) {
		int N = data.length;
		double[][] newData = null;
		switch(k) {
			case 3: newData = new double[N][4]; break;
			case 4: newData = new double[N][5]; break;
			case 5: newData = new double[N][6]; break;
			case 6: newData = new double[N][7]; break;
			case 7: newData = new double[N][8]; break;
		}
		for(int i = 0; i < N; i++) {
			newData[i][0] = data[i][0];
			newData[i][1] = data[i][1];
			newData[i][2] = data[i][2];
			newData[i][3] = data[i][1]*data[i][1];
			if(k >= 4) {
				newData[i][4] = data[i][2]*data[i][2];
			}
			if(k >= 5) {
				newData[i][5] = data[i][1]*data[i][2];
			}
			if(k >= 6) {
				newData[i][6] = Math.abs(data[i][1] - data[i][2]);
			}
			if(k == 7) {
				newData[i][7] = Math.abs(data[i][1] + data[i][2]);
			}
		}
		return newData;
	}
}