import java.io.*;
import java.util.ArrayList;

class DigitClassifier {

	public int train_size = 0;
	public int test_size = 0;
	public ArrayList<Digit> train_data = new ArrayList<>();
	public ArrayList<Digit> test_data = new ArrayList<>();
	public Model model;
	int one = -1;
	int two = -1;
	boolean is_transformed;
	
	public DigitClassifier(String train_data_file, String test_data_file) throws IOException {
		read_data(train_data_file, test_data_file);
	}
	
	public void read_data(String train_data_file, String test_data_file) throws IOException {
		String line = null;
		String[] tokens = null;
		BufferedReader input = new BufferedReader(new FileReader(train_data_file));
		while((line = input.readLine()) != null) {
			tokens = line.split("\\s+");
			int digit = (int) Double.parseDouble(tokens[0]);
			train_data.add(new Digit(digit, Double.parseDouble(tokens[1]), Double.parseDouble(tokens[2])));
			train_size++;
		}
		input.close();
		input = new BufferedReader(new FileReader(test_data_file));
		while((line = input.readLine()) != null) {
			tokens = line.split("\\s+");
			int digit = (int) Double.parseDouble(tokens[0]);
			test_data.add(new Digit(digit, Double.parseDouble(tokens[1]), Double.parseDouble(tokens[2])));
			test_size++;
		}
		input.close();
	}
	
	public void train_one_vs_all(int one, boolean is_transformed, double lambda) {
		this.is_transformed = is_transformed;
		this.one = one;
		this.two = -1;
		double[][] training_set = new double[train_size][3];
		int[] labels = new int[train_size];
		for(int i = 0; i < train_size; i++) {
			Digit digit = train_data.get(i);
			training_set[i][0] = 1;
			training_set[i][1] = digit.symmetry;
			training_set[i][2] = digit.intensity;
			if(digit.digit == one) {
				labels[i] = 1;
			} else {
				labels[i] = -1;
			}
		}
		if(is_transformed) {
			training_set = Quadratic2DTransformer.transform(training_set);
			model = LinearRegression.get_model(training_set, labels, lambda);
		} else {
			model = LinearRegression.get_model(training_set, labels, lambda);
		}
	}
	
	public void train_one_vs_one(int one, int two, boolean is_transformed, double lambda) {
		this.is_transformed = is_transformed;
		this.one = one;
		this.two = two;
		ArrayList<double[]> training_set = new ArrayList<>();
		ArrayList<Integer> labels = new ArrayList<>();
		for(int i = 0; i < train_size; i++) {
			Digit digit = train_data.get(i);
			if(digit.digit == one || digit.digit == two) {
				training_set.add(new double[]{1, digit.symmetry, digit.intensity});
				if(digit.digit == one) {
					labels.add(1);
				} else {
					labels.add(-1);
				}
			}
		}
		double[][] data = new double[training_set.size()][];
		for(int i = 0; i < data.length; i++) {
			data[i] = training_set.get(i);
		}
		int[] labs = new int[data.length];
		for(int i = 0; i < data.length; i++) {
			labs[i] = labels.get(i);
		}
		if(is_transformed) {
			model = LinearRegression.get_model(Quadratic2DTransformer.transform(data), labs, lambda);
		} else {
			model = LinearRegression.get_model(data, labs, lambda);
		}
	}
	
	public int classify(Digit digit) {
		if(is_transformed) {
			return  model.classify(Quadratic2DTransformer.transform(new double[]{1, digit.symmetry, digit.intensity}));
		}
		return model.classify(new double[]{1, digit.symmetry, digit.intensity});
	}
	
	public double E_in_one_vs_all() {
		int incorrect = 0;
		for(Digit d : train_data) {
			int label = classify(d);
			if((label == 1 && d.digit != one) || (label == -1 && d.digit == one)) {
				incorrect++;
			}
		}
		return (incorrect + 0.0)/train_size;
	}	
	
	public double E_out_one_vs_all() {
		int incorrect = 0;
		for(Digit d : test_data) {
			int label = classify(d);
			if((label == 1 && d.digit != one) || (label == -1 && d.digit == one)) {
				incorrect++;
			}
		}
		return (incorrect + 0.0)/train_size;
	}
	
	public double E_in_one_vs_one() {
		int incorrect = 0;
		int count = 0;
		for(Digit d : train_data) {
			if(d.digit == one || d.digit == two) {
				int label = classify(d);
				if((label == 1 && d.digit == two) || (label == -1 && d.digit == one)) {
					incorrect++;
				}
				count++;
			}
		}
		return (incorrect + 0.0)/count;
	}	
	
	public double E_out_one_vs_one() {
		int incorrect = 0;
		int count = 0;
		for(Digit d : test_data) {
			if(d.digit == one || d.digit == two) {
				int label = classify(d);
				if((label == 1 && d.digit == two) || (label == -1 && d.digit == one)) {
					incorrect++;
				}
				count++;
			}
		}
		return (incorrect + 0.0)/train_size;
	}

	public static void test_one_vs_all(String train, String test) throws IOException {
		DigitClassifier dc = new DigitClassifier(train, test);
		for(int i = 0; i < 10; i++) {
			dc.train_one_vs_all(i, false, 1);
			StdOut.println(String.format(java.util.Locale.UK, "\nFor " + i + " without transform: E_in = %.4f E_out = %.4f", dc.E_in_one_vs_all(), dc.E_out_one_vs_all()));
			dc.train_one_vs_all(i, true, 1);
			StdOut.println(String.format(java.util.Locale.UK, "For " + i + " with    transform: E_in = %.4f E_out = %.4f", dc.E_in_one_vs_all(), dc.E_out_one_vs_all()));
		}
	}
	
	public static void test_one_vs_one(String train, String test) throws IOException {
		DigitClassifier dc = new DigitClassifier(train, test);
		dc.train_one_vs_one(1, 5, true, 0.01);
		StdOut.println(String.format(java.util.Locale.UK, "\nFor lambda = " + 0.01 + " E_in = %.4f E_out = %.4f", dc.E_in_one_vs_one(), dc.E_out_one_vs_one()));
		dc.train_one_vs_one(1, 5, true, 1);
		StdOut.println(String.format(java.util.Locale.UK, "For lambda = " + 1 + "    E_in = %.4f E_out = %.4f", dc.E_in_one_vs_one(), dc.E_out_one_vs_one()));
	}
	
	public static void main(String[] args) throws IOException {
		/* int one = Integer.parseInt(args[2]);
		double lambda = Double.parseDouble(args[3]);
		boolean is_transformed = Boolean.parseBoolean(args[4]);
		DigitClassifier dc = new DigitClassifier(args[0], args[1]);
		dc.train_one_vs_all(one, is_transformed, lambda); 
		StdOut.println("E_in = " + dc.E_in_one_vs_all());
		StdOut.println("E_out = " + dc.E_out_one_vs_all()); */ 
		//test_one_vs_all(args[0], args[1]);
		test_one_vs_one(args[0], args[1]);
	}
}