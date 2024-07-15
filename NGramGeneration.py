from collections import defaultdict
from nltk import ngrams

def generate_ngrams(word_sequence, pos_tag_sequence, lemma_sequence, ngram_range=(2, 7), add_newline=False):
    ngram_sequences = defaultdict(list)
    
    # Convert strings to lists
    words = word_sequence.split()
    pos_tags = pos_tag_sequence.split()
    lemmas = lemma_sequence.split()
    
    for n in range(ngram_range[0], ngram_range[1] + 1):
        pos_n_grams = list(ngrams(pos_tags, n))
        word_n_grams = list(ngrams(words, n))
        lemma_n_grams = list(ngrams(lemmas, n))
        
        for pos_gram, word_gram, lemma_gram in zip(pos_n_grams, word_n_grams, lemma_n_grams):
            unique_tags = set(pos_gram)
            if len(unique_tags) >= 4:
                ngram_str = ' '.join(word_gram)
                lemma_str = ' '.join(lemma_gram)
                pos_str = ' '.join(pos_gram)
                if add_newline:
                    ngram_str += '\n'
                    lemma_str += '\n'
                    pos_str += '\n'
                ngram_sequences[n].append((ngram_str, pos_str, lemma_str))
    
    return ngram_sequences

# Example usage
if __name__ == "__main__":
    # Example input sequences as strings with punctuations
    word_sequence = "Si Juan ay gusto kumain ng pizza, at si Maria ay kumain ng cake."
    pos_tag_sequence = "NNP VBZ TO VB NN PUNCT CONJ NNP VBZ VB NN PUNCT"
    lemma_sequence = "si juan ay gusto kain ng pizza , at si maria ay kain ng cake ."
    
    # Generate n-grams with default parameters
    ngram_sequences = generate_ngrams(word_sequence, pos_tag_sequence, lemma_sequence)
    
    # Print the generated n-grams
    for n, ngrams_list in ngram_sequences.items():
        print(f"Generated {n}-grams:")
        for ngram in ngrams_list:
            ngram_str, pos_str, lemma_str = ngram
            print(f"Words: {ngram_str}\nPOS: {pos_str}\nLemmas: {lemma_str}\n")
        print()