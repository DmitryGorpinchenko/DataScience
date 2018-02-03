class Coin {
	private int status;
	
	public Coin() {
		flip();
	}
	
	public boolean is_head() {
		return (status == 1);
	}
	
	public void flip() {
		status = (int) (2*Math.random());
	}
}