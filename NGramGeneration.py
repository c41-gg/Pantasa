import csv
from collections import defaultdict
from nltk import ngrams

def generate_ngrams(word_sequence, rough_pos_sequence, detailed_pos_sequence, lemma_sequence, ngram_range=(2, 7), add_newline=False, start_id=0):
    ngram_sequences = defaultdict(list)
    
    # Convert strings to lists
    words = word_sequence.split()
    rough_pos_tags = rough_pos_sequence.split()
    detailed_pos_tags = detailed_pos_sequence.split()
    lemmas = lemma_sequence.split()
    
    current_id = start_id
    
    for n in range(ngram_range[0], ngram_range[1] + 1):
        rough_pos_n_grams = list(ngrams(rough_pos_tags, n))
        detailed_pos_n_grams = list(ngrams(detailed_pos_tags, n))
        word_n_grams = list(ngrams(words, n))
        lemma_n_grams = list(ngrams(lemmas, n))
        
        for rough_pos_gram, detailed_pos_gram, word_gram, lemma_gram in zip(rough_pos_n_grams, detailed_pos_n_grams, word_n_grams, lemma_n_grams):
            # Split combined detailed POS tags and add to unique detailed tags set
            unique_detailed_tags = set(tag for detailed_tag in detailed_pos_gram for tag in detailed_tag.split('_'))
            
            if len(unique_detailed_tags) >= 4:
                ngram_str = ' '.join(word_gram)
                lemma_str = ' '.join(lemma_gram)
                rough_pos_str = ' '.join(rough_pos_gram)
                detailed_pos_str = ' '.join(detailed_pos_gram)
                if add_newline:
                    ngram_str += '\n'
                    lemma_str += '\n'
                    rough_pos_str += '\n'
                    detailed_pos_str += '\n'
                ngram_id = f"{current_id:06d}"
                ngram_sequences[n].append((ngram_id, n, rough_pos_str, detailed_pos_str, ngram_str, lemma_str))
                current_id += 1
    
    return ngram_sequences, current_id

def process_csv(input_file, output_file):
    results = []
    start_id = 0

    with open(input_file, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        
        for row in reader:
            sentence = row['Sentences']
            rough_pos = row['General_POS_Tagged']
            detailed_pos = row['Detailed_POS_Tagged']
            lemmatized = row['Lemmatized']
            
            ngram_data, start_id = generate_ngrams(sentence, rough_pos, detailed_pos, lemmatized, start_id=start_id)
            
            for ngram_size, ngrams_list in ngram_data.items():
                for ngram_tuple in ngrams_list:
                    ngram_id, ngram_size, rough_pos_str, detailed_pos_str, ngram_str, lemma_str = ngram_tuple
                    results.append({
                        'N-Gram_ID': ngram_id,
                        'N-Gram_Size': ngram_size,
                        'RoughPOS_N-Gram': rough_pos_str,
                        'DetailedPOS_N-Gram': detailed_pos_str,
                        'N-Gram': ngram_str,
                        'Lemma_N-Gram': lemma_str
                    })
    
    # Write results to output CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
        fieldnames = ['N-Gram_ID', 'N-Gram_Size', 'RoughPOS_N-Gram', 'DetailedPOS_N-Gram', 'N-Gram', 'Lemma_N-Gram']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

# Example usage
input_csv = 'database/preprocessed.csv'
output_csv = 'database/ngrams.csv'
process_csv(input_csv, output_csv)
