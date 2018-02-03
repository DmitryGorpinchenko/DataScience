class Sentence {
	
	public int count;
	public String text;
	
	public Sentence(String text, int count) {
		this.text = text;
		this.count = count;
	}
	
	public int hashCode() {
		return text.hashCode();
	}
	
	public boolean equals(Sentence that) {
		return this.text.equals(that.text);
	}
	
	public String toString() {
		return count + " " + text;
	}
}