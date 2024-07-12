import csv
from collections import defaultdict

# Function to generate n-grams
def generate_ngrams(tokens, n):
    ngrams = [tokens[i:i+n] for i in range(len(tokens)-n+1)]
    return [" ".join(ngram) for ngram in ngrams]

# Function to cluster n-grams
def cluster_ngrams(tagged_sentences, min_n=4):
    clustered_ngrams = defaultdict(list)
    for sentence in tagged_sentences:
        tokens = sentence.split()
        for n in range(min_n, len(tokens) + 1):  # Generate n-grams from size min_n to the length of the tokens
            ngrams = generate_ngrams(tokens, n)
            clustered_ngrams[n].extend(ngrams)
    return clustered_ngrams


from collections import defaultdict

# Function to cluster n-grams
def cluster_ngrams(ngrams, min_n=4):
    clustered_ngrams = defaultdict(list)
    for ngram in ngrams:
        tokens = ngram.split()
        n = len(tokens)
        if n >= min_n:
            clustered_ngrams[n].append(ngram)
    return clustered_ngrams

# Example usage
if __name__ == "__main__":
    # Path to the CSV file containing the pre-generated n-grams
    input_csv = "ngram_sentences.csv"

    # Read n-grams from the CSV file
    ngrams = ["Ang mga tao ay may iba't ibang wika at kultura" , "Ang BERT model ay ginagamit para sa natural language processing.", "Si Juan ay mahilig magbasa ng mga libro sa kanyang libreng oras.",  "Ang Pilipinas ay isang arkipelago na matatagpuan sa Timog-Silangang Asya.", "Ang RoBERTa ay isang variant ng BERT na mas mahusay sa ilang mga task."]


    # Cluster n-grams
    clustered_ngrams = cluster_ngrams(ngrams)

    # Output the clusters to the terminal
    for n in sorted(clustered_ngrams):
        print(f"\nClustered {n}-grams:\n")
        for ngram in clustered_ngrams[n]:
            print(ngram)
 