import csv
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import os

# Define the model
tagalog_roberta_model = "jcblaise/roberta-tagalog-base"
print("Loading tokenizer...")
roberta_tokenizer = AutoTokenizer.from_pretrained(tagalog_roberta_model)
print("Loading model...")
roberta_model = AutoModelForMaskedLM.from_pretrained(tagalog_roberta_model)
print("Model and tokenizer loaded successfully.")

def load_csv(file_path):
    # Load data from a CSV file into a list of dictionaries.
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    print(f"Data loaded from {file_path}. Number of rows: {len(data)}")
    return data

def convert_id_array(id_array_str):
    return id_array_str.strip("[]'").replace("'", "").split(', ')

def load_and_convert_csv(file_path):
    data = load_csv(file_path)
    for entry in data:
        entry['ID_Array'] = convert_id_array(entry['ID_Array'])
    return data

def compute_mlm_score(sentence, model, tokenizer):
    # Compute MLM (Masked Language Model) score for a given sentence.
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    input_ids = inputs['input_ids']
    probs = torch.softmax(logits, dim=-1)
    token_probs = torch.gather(probs, 2, input_ids.unsqueeze(-1)).squeeze(-1).tolist()
    
    # Flatten the list if it's nested
    if isinstance(token_probs[0], list):
        token_probs = [item for sublist in token_probs for item in sublist]
    
    # Normalize the probabilities to balance the scores
    min_prob = min(token_probs)
    max_prob = max(token_probs)
    token_probs = [(score - min_prob) / (max_prob - min_prob) * 100 for score in token_probs]
    
    return sum(token_probs) / len(token_probs)  # Return the average score as the score for the whole sequence

def pattern_exists(pattern, results, key):
    return any(result[key] == pattern for result in results)

def load_existing_results(output_file):
    if not os.path.exists(output_file):
        return []

    with open(output_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row['Final_Hybrid_N-Gram'] for row in reader]

def generalize_patterns(ngram_list_file, pos_patterns_file, id_array_file, output_file, model, tokenizer, threshold=80.0):
    # Generalize POS tag patterns with corresponding word and lemma ngrams based on MLM scores.

    print("Loading ngram list...")
    ngram_list = load_csv(ngram_list_file)
    print("Loading POS patterns...")
    pos_patterns = load_csv(pos_patterns_file)
    print("Loading ID array data...")
    id_array_data = load_and_convert_csv(id_array_file)
    
    pos_comparison_results = []

    # Load existing results from the output file to avoid duplicates
    existing_hybrid_ngrams = load_existing_results(output_file)

    # Create a dictionary for quick access to POS patterns by Pattern_ID
    pos_patterns_dict = {entry['Pattern_ID']: entry['POS_N-Gram'] for entry in pos_patterns}

    # Iterate through each entry in the id_array_data
    for id_array_index, id_array_entry in enumerate(id_array_data):
        print(f"Processing ID array entry {id_array_index + 1}/{len(id_array_data)}...")
        pattern_id = id_array_entry['Pattern_ID']
        
        # Retrieve the POS pattern for the current pattern ID from the POS patterns dictionary
        rough_pos = pos_patterns_dict.get(pattern_id, None)
        if not rough_pos:
            print(f"No POS pattern found for Pattern_ID {pattern_id}. Skipping...")
            continue

        # Check if the original POS N-Gram is already in the results
        if not pattern_exists(rough_pos, pos_comparison_results, 'POS_N-Gram'):
            pos_comparison_results.append({
                'Pattern_ID': pattern_id,
                'POS_N-Gram': rough_pos,
                'Lexeme_N-Gram': '',
                'MLM_Scores': [],
                'Comparison_Replacement_Matrix': '',
                'Final_Hybrid_N-Gram': rough_pos  # Set the default hybrid n-gram to the original POS n-gram
            })

        # Iterate through each instance ID in the ID array
        for instance_index, instance_id in enumerate(id_array_entry['ID_Array']):
            # Convert instance_id to 6-digit number format
            instance_id = instance_id.zfill(6)
            print(f"Processing instance {instance_index + 1}/{len(id_array_entry['ID_Array'])} for pattern ID {pattern_id}...")

            # Retrieve corresponding instance from ngram_list using instance_id
            instance = next((ngram for ngram in ngram_list if ngram['N-Gram_ID'] == instance_id), None)
            if not instance:
                print(f"No instance found for ID {instance_id}.")
                continue

            word_ngram_sentence = instance['N-Gram']
            lemma_ngram_sentence = instance['Lemma_N-Gram']

            for ngram_sentence, ngram_type in [(word_ngram_sentence, 'word'), (lemma_ngram_sentence, 'lemma')]:
                if not ngram_sentence:
                    print(f"No {ngram_type} ngram sentence found for instance ID {instance_id}. Skipping...")
                    continue

                print(f"Computing MLM score for {ngram_type} ngram sentence: {ngram_sentence}...")
                # Compute MLM score for the whole ngram sentence
                sequence_mlm_score = compute_mlm_score(ngram_sentence, model, tokenizer)
                
                # Debug statements to check the sequence score
                print(f"Sequence MLM score: {sequence_mlm_score}")

                # Check if sequence MLM score meets the threshold for further processing
                if sequence_mlm_score >= threshold:
                    print(f"Sequence MLM score {sequence_mlm_score} meets the threshold {threshold}. Computing individual word scores...")
                    comparison_matrix = ['*'] * len(ngram_sentence.split())
                    new_pattern = rough_pos.split()

                    words = ngram_sentence.split()
                    rough_pos_tokens = rough_pos.split()

                    # Ensure the lengths match to avoid index errors
                    if len(words) != len(rough_pos_tokens):
                        print(f"Length mismatch between words and POS tokens for instance ID {instance_id}. Skipping...")
                        continue

                    # Iterate through each word in the ngram sentence
                    for i, (pos_tag, word) in enumerate(zip(rough_pos_tokens, words)):
                        word_score = compute_mlm_score(word, model, tokenizer)
                        print(f"Word '{word}' score: {word_score}")  # Debug statement for word score
                        if word_score >= threshold:
                            new_pattern[i] = word
                            comparison_matrix[i] = word

                    final_hybrid_ngram = ' '.join(new_pattern)

                    # Check if the final hybrid ngram already exists
                    if final_hybrid_ngram not in existing_hybrid_ngrams:
                        lexeme_ngram = ngram_sentence  # Only use the current ngram sentence (word or lemma)

                        # Prepare result dictionary and append to results list
                        pos_comparison_results.append({
                            'Pattern_ID': pattern_id,
                            'POS_N-Gram': rough_pos,
                            'Lexeme_N-Gram': lexeme_ngram,
                            'MLM_Scores': sequence_mlm_score,
                            'Comparison_Replacement_Matrix': ' '.join(comparison_matrix),
                            'Final_Hybrid_N-Gram': final_hybrid_ngram
                        })
                        existing_hybrid_ngrams.append(final_hybrid_ngram)  # Add to existing hybrid ngrams

    # Write results to output file
    print(f"Writing results to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['Pattern_ID', 'POS_N-Gram', 'Lexeme_N-Gram', 'MLM_Scores', 'Comparison_Replacement_Matrix', 'Final_Hybrid_N-Gram']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in pos_comparison_results:
            writer.writerow(result)

    # Print status progress
    print(f"Generalization completed. Results written to {output_file}")

# Example usage
for n in range(2, 8):
    ngram_list_file = 'database/ngrams.csv'
    pos_patterns_file = f'database/Generalized/POSTComparison/{n}grams.csv'
    id_array_file =  f'database/POS/{n}grams.csv'
    output_file = f'database/Generalized/LexemeComparison/{n}grams.csv'

    print(f"Starting generalization for {n}-grams...")
    generalize_patterns(ngram_list_file, pos_patterns_file, id_array_file, output_file, roberta_model, roberta_tokenizer)
    print(f"Finished generalization for {n}-grams.")
