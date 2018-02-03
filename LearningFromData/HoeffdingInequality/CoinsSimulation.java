class CoinsSimulation {
	private Coin[] coins;
	private double first_coin_head_freq;
	private double rand_coin_head_freq;
	private double min_coin_head_freq;
	private int coins_number;
	private int experiments_number;
	private int trials_per_experiment;
	private double run_time;
	
	public CoinsSimulation(int coins_number, int experiments_number, int trials_per_experiment) {
		this.coins_number = coins_number;
		this.experiments_number = experiments_number;
		this.trials_per_experiment = trials_per_experiment;
		coins = new Coin[coins_number];
		for(int i = 0; i < coins_number; i++) {
			coins[i] = new Coin();
		}
	}
	
	public void run_simulation() {
		Stopwatch sw = new Stopwatch();
		for(int i = 0; i < experiments_number; i++) {
			int[] head_counts = new int[coins_number];
			for(int j = 0; j < coins_number; j++) {
				for(int k = 0; k < trials_per_experiment; k++) {
					coins[j].flip();
					if(coins[j].is_head()) {
						head_counts[j]++;
					}
				}
			}
			first_coin_head_freq += (head_counts[0] + 0.0)/trials_per_experiment;
			rand_coin_head_freq += (head_counts[StdRandom.uniform(0, head_counts.length)] + 0.0)/trials_per_experiment;
			min_coin_head_freq += (min(head_counts) + 0.0)/trials_per_experiment;
		}
		first_coin_head_freq = first_coin_head_freq/experiments_number;
		rand_coin_head_freq = rand_coin_head_freq/experiments_number;
		min_coin_head_freq = min_coin_head_freq/experiments_number;
		run_time = sw.elapsedTime();
	}
	
	public void print_results() {
		StdOut.println("Average heads freq for the first coin: " + first_coin_head_freq);
		StdOut.println("Average heads freq for the random coin: " + rand_coin_head_freq);
		StdOut.println("Average heads freq for the min-head coin: " + min_coin_head_freq);
		StdOut.println("\nSimulation running time: " + run_time);
	}
	
	public int min(int[] a) {
		int min = Integer.MAX_VALUE;
		for(int i = 0; i < a.length; i++) {
			if(a[i] < min) {
				min = a[i];
			}
		}
		return min;
	}
	
	public static void main(String[] args) {
		int coins_number = Integer.parseInt(args[0]);
		int experiments_number = Integer.parseInt(args[1]);
		int trials_per_experiment = Integer.parseInt(args[2]);
		CoinsSimulation sim = new CoinsSimulation(coins_number, experiments_number, trials_per_experiment);
		sim.run_simulation();
		sim.print_results();
	}
}