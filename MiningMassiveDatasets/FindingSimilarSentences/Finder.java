import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.io.*;

class Finder {

	private int similar_pairs = 0;
	
	public Finder(String file, boolean is_preprocess) throws IOException {
		if(is_preprocess) {
			preprocess(file);
		}
		String line = null;
		BufferedReader input =  new BufferedReader(new FileReader("lengths.txt"));
		ArrayList<Integer> lengths = new ArrayList<>();
		while((line = input.readLine()) != null) {
			lengths.add(Integer.parseInt(line));
		}
		input.close();
		run_pass1(lengths);
		run_pass2(lengths);
	}
	
	public void preprocess(String file) throws IOException {
		Preprocessor.split_by_length(file);
	}
	
	public void run_pass1(ArrayList<Integer> lengths) throws IOException {
		StdOut.println("\nCounting pairs with substitution dissimilarity ...\n");
		String line = null;
		String[] words;
		for(int len : lengths) {
			StdOut.println("Length " + len + " ...");
			HashMap<String, HashSet<Sentence>> index = new HashMap<>();
			BufferedReader input = new BufferedReader(new FileReader("data/" + len + ".txt"));
			while(true) {
				line = input.readLine();
				if(line == null) {
					break;
				}
				int id = line.indexOf(" ");
				String text = line.substring(id+1);
				int count = Integer.parseInt(line.substring(0, id));	
				similar_pairs += count*(count-1)/2; //counting identical pairs
				put_in_index(new Sentence(text, count), index);
			}				
			input.close();
			similar_pairs += get_similar_pairs_num(index);
		}
	}
	
	public void run_pass2(ArrayList<Integer> lengths) throws IOException {
		StdOut.println("\nCounting pairs with deletion/insertion dissimilarity ...\n");
		String line = null;
		int size = lengths.size();
		for(int i = 0; i < size-1; i++) {
			int prev = lengths.get(i);
			int curr = lengths.get(i+1);
			if(prev != (curr - 1)) {
				continue;
			}
			StdOut.print(prev + " ");
			BufferedReader prev_input = new BufferedReader(new FileReader("data/" + prev + ".txt"));
			BufferedReader curr_input = new BufferedReader(new FileReader("data/" + curr + ".txt"));
			HashMap<String, HashSet<Sentence>> index = new HashMap<>();
			HashMap<String, Integer> counts = new HashMap<>();
			while(true) {
				line = prev_input.readLine();
				if(line == null) {
					break;
				}
				int id = line.indexOf(" ");
				String text = line.substring(id+1);
				int count = Integer.parseInt(line.substring(0, id));
				counts.put(text, count);
				index.put(text, new HashSet<Sentence>());
			}
			while(true) {
				line = curr_input.readLine();
				if(line == null) {
					break;
				}
				int id = line.indexOf(" ");
				String text = line.substring(id+1);
				int count = Integer.parseInt(line.substring(0, id));
				Sentence sentence = new Sentence(text, count);
				id = -1;
				String begin = "";
				String temp = null;
				while(true)	{
					id = text.indexOf(" ", id+1);
					if(id == -1) {
						temp = begin.substring(0, begin.length()-1);
					} else {
						temp = begin + text.substring(id + 1);
					}
					begin = text.substring(0, id+1);
					if(index.containsKey(temp)) {
						index.get(temp).add(sentence);
					}
					if(id == -1) {
						break;
					}
				}
			}
			prev_input.close();
			curr_input.close();
			for(String s : index.keySet()) {
				int count = counts.get(s);
				for(Sentence sent : index.get(s)) {
					similar_pairs += count*sent.count;
				}
			}
		}
		StdOut.println();
	}
	
	public int get_pairs_num() {
		return similar_pairs;
	}
	
	public void put_in_index(Sentence sent, HashMap<String, HashSet<Sentence>> index) {
		String[] words = sent.text.split(" ");
		int len = words.length;
		StringBuffer part1_ = new StringBuffer(words[0]);
		StringBuffer part2_ = new StringBuffer(words[len/2]);
		for(int i = 1; i < len/2; i++) {
			part1_.append(" " + words[i]);
		}
		for(int i = len/2+1; i < len; i++) {
			part2_.append(" " + words[i]);
		}
		String part1 = part1_.toString();
		String part2 = part2_.toString();
		if(index.containsKey(part1)) {
			index.get(part1).add(sent);
		} else {
			index.put(part1, new HashSet<Sentence>());
			index.get(part1).add(sent);
		}
		if(index.containsKey(part2)) {
			index.get(part2).add(sent);
		} else {
			index.put(part2, new HashSet<Sentence>());
			index.get(part2).add(sent);
		} 
	}
	
	public int get_similar_pairs_num(HashMap<String, HashSet<Sentence>> index) {
		int count = 0;
		for(String s : index.keySet()) {
			ArrayList<Sentence> sentences = new ArrayList<>();
			for(Sentence sentence : index.get(s)) {
				sentences.add(sentence);
			}
			if(sentences.size() == 1) {
				continue;
			}
			int n = sentences.size();
			for(int i = 0; i < n-1; i++) {
				for(int j = i+1; j < n; j++) {
					Sentence s1 = sentences.get(i);
					Sentence s2 = sentences.get(j);
					if(is_similar(s1.text.split(" "), s2.text.split(" "))) {
						count += s1.count*s2.count;
					}
				}
			}
		}
		return count;
	}

	public static boolean is_similar(String[] word1, String[] word2) {
		int count = 0;
		for(int i = 0; i < word1.length; i++) {
			if(!word1[i].equals(word2[i])) {
				count++;
			}
			if(count > 1) {
				break;
			}
		}
		return (count == 1);
	}
	
	public static void main(String[] args) throws IOException {
		Stopwatch sw = new Stopwatch();
		String file = args[0];
		boolean is_preprocess = false;
		if("-p".equals(args[1]) && "true".equals(args[2])) {
			is_preprocess = true;
		}
		Finder f = new Finder(file, is_preprocess);
		StdOut.println("\nSimilar pairs num = " + f.get_pairs_num());
		StdOut.println("\nTiming results: " + sw.elapsedTime() + " seconds");  		
	}
}