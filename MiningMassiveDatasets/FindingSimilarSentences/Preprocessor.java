import java.io.*;
import java.util.HashMap;
import java.util.TreeSet;

class Preprocessor {
	
	public static void split_by_length(String file) throws IOException {
		BufferedReader input =  new BufferedReader(new FileReader(file));
		HashMap<String, Integer> sentences = new HashMap<>();
		TreeSet<Integer> lengths = new TreeSet<Integer>();
		String line = null;
		int count = 0;
		StdOut.println("\nCounting duplicates...\n");
		while((line = input.readLine()) != null) {
			String sent = line.substring(line.indexOf(" ") + 1);
			lengths.add(sent.split(" ").length);
			if(sentences.containsKey(sent)) {
				sentences.put(sent, sentences.get(sent) + 1);
			} else {
				sentences.put(sent, 1);
			}
			count++;
			if(count%100000 == 0) {
				StdOut.println(count);
			}
		}
		input.close();
		StdOut.println("\nWriting different sentence lengths into file ...\n");
		HashMap<Integer, BufferedWriter> files = new HashMap<>();
		BufferedWriter output =  new BufferedWriter(new FileWriter("lengths.txt"));
		for(int l : lengths) {
			output.write(l + "\n");
			files.put(l, new BufferedWriter(new FileWriter("data/" + l + ".txt")));
		}
		output.close();
		StdOut.println("\nSplitting by lengths ...\n");
		count = 0;
		for(String s : sentences.keySet()) {
			count++;
			if(count%100000 == 0) {
				StdOut.println(count);
			}
			files.get(s.split(" ").length).write(sentences.get(s) + " " + s + "\n");
		}
		for(int l : files.keySet()) {
			files.get(l).close();
		}
	}
}