import csv
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import os

# Define the model
global_max_score = 0
global_min_score = 0
tagalog_roberta_model = "jcblaise/roberta-tagalog-base"
print("Loading tokenizer...")
roberta_tokenizer = AutoTokenizer.from_pretrained(tagalog_roberta_model)
print("Loading model...")
roberta_model = AutoModelForMaskedLM.from_pretrained(tagalog_roberta_model)
print("Model and tokenizer loaded successfully.")

def load_csv(file_path):
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

def print_tokenization(sentence, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    print(f"Sentence: {sentence}")
    print(f"Tokens: {tokens}")
    return tokens

def print_model_outputs(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    print(f"Logits: {logits}")
    print(f"Probabilities: {probs}")

def compute_mlm_score(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt")
    print_tokenization(sentence, tokenizer)
    print_model_outputs(sentence, model, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    input_ids = inputs['input_ids'].squeeze()
    probs = torch.softmax(logits, dim=-1).squeeze()

    # Compute average probability for each token
    scores = []
    min_score = float('inf')
    max_score = float('-inf')
    for i, token_id in enumerate(input_ids):
        token_prob = probs[i, token_id].item()
        scores.append(token_prob)
        if token_prob > max_score:
            max_score = token_prob
        if token_prob < min_score:
            min_score = token_prob

    average_score = sum(scores) / len(scores)
    return average_score * 100, min_score, max_score, scores  # Return both average score and scores array

def compute_word_scores(word, sentence, model, tokenizer):
    # Tokenize the full sentence
    inputs = tokenizer(sentence, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    input_ids = inputs['input_ids'].squeeze()
    probs = torch.softmax(logits, dim=-1).squeeze()

    # Tokenize the word separately
    word_tokens = tokenizer.tokenize(word)
    word_token_ids = tokenizer.convert_tokens_to_ids(word_tokens)

    # Tokenize the full sentence and get token IDs
    sentence_tokens = tokenizer.tokenize(sentence)
    sentence_token_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
    
    # Compute MLM score for the sentence and get the score array
    sentence_mlm_score, min_score, max_score, score_array = compute_mlm_score(sentence, model, tokenizer)
    
    # Map token IDs to their MLM scores
    token_id_to_prob = {token_id: score_array[i] for i, token_id in enumerate(sentence_token_ids)}

    # Compute contextual probability for each token in the word
    word_scores = []
    for token in word_tokens:
        # Convert token to ID
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id in token_id_to_prob:
            token_prob = token_id_to_prob[token_id]
            word_scores.append(token_prob)
        else:
            # Handle tokens with the "Ġ" prefix
            token_with_prefix = f"Ġ{token}"
            token_id_with_prefix = tokenizer.convert_tokens_to_ids(token_with_prefix)
            if token_id_with_prefix in token_id_to_prob:
                token_prob = token_id_to_prob[token_id_with_prefix]
                word_scores.append(token_prob)
            else:
                print(f"Token ID {token_id} or {token_id_with_prefix} for '{token}' not found in token_id_to_prob.")

    # Print debugging information
    print(f"Word tokens: {word_tokens}")
    print(f"Word token IDs: {word_token_ids}")
    print(f"Sentence tokens: {sentence_tokens}")
    print(f"Sentence token IDs: {sentence_token_ids}")
    print(f"Token ID to Probability Mapping: {token_id_to_prob}")
    print(f"Word Scores: {word_scores}")

    # Normalize the scores
    if word_scores:
        if max_score > min_score:
            normalized_scores = [(score - min_score) / (max_score - min_score) for score in word_scores]
        else:
            normalized_scores = [0.5] * len(word_scores)
        
        average_normalized_score = sum(normalized_scores) / len(normalized_scores) * 100  # Convert to percentage
    else:
        average_normalized_score = 0

    return average_normalized_score

def load_existing_results(output_file):
    if not os.path.exists(output_file):
        return set()

    with open(output_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        existing_ngrams = {row['Final_Hybrid_N-Gram'] for row in reader}
    return existing_ngrams

def pattern_exists(pos_ngram, pos_comparison_results, key):
    return any(result[key] == pos_ngram for result in pos_comparison_results)

def get_latest_pattern_id(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            pattern_ids = [int(row['Pattern_ID']) for row in reader if row['Pattern_ID'].isdigit()]
            return max(pattern_ids, default=0)
    except FileNotFoundError:
        return 0
    
def generate_pattern_id(counter):
    return f"{counter:06d}"

def generalize_patterns(ngram_list_file, pos_patterns_file, id_array_file, output_file, model, tokenizer, threshold=80.0):
    print("Loading ngram list...")
    ngram_list = load_csv(ngram_list_file)
    print("Loading POS patterns...")
    pos_patterns = load_csv(pos_patterns_file)
    print("Loading ID array data...")
    id_array_data = load_and_convert_csv(id_array_file)
    
    latest_pattern_id_input = get_latest_pattern_id(pos_patterns_file)
    latest_pattern_id_output = get_latest_pattern_id(output_file)
    latest_pattern_id = max(latest_pattern_id_input, latest_pattern_id_output)
    pattern_counter = latest_pattern_id + 1

    
    pos_comparison_results = []
    existing_hybrid_ngrams = load_existing_results(output_file)
    pos_patterns_dict = {entry['Pattern_ID']: entry['POS_N-Gram'] for entry in pos_patterns}

    for id_array_index, id_array_entry in enumerate(id_array_data):
        print(f"Processing ID array entry {id_array_index + 1}/{len(id_array_data)}...")
        pattern_id = id_array_entry['Pattern_ID']
        rough_pos = pos_patterns_dict.get(pattern_id, None)
        if not rough_pos:
            print(f"No POS pattern found for Pattern_ID {pattern_id}. Skipping...")
            continue

        if not pattern_exists(rough_pos, pos_comparison_results, 'POS_N-Gram'):
            pos_comparison_results.append({
                'Pattern_ID': pattern_id,
                'POS_N-Gram': rough_pos,
                'Lexeme_N-Gram': '',
                'MLM_Scores': [],
                'Comparison_Replacement_Matrix': '',
                'Final_Hybrid_N-Gram': rough_pos
            })

        for instance_index, instance_id in enumerate(id_array_entry['ID_Array']):
            instance_id = instance_id.zfill(6)
            print(f"Processing instance {instance_index + 1}/{len(id_array_entry['ID_Array'])} for pattern ID {pattern_id}...")

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
                initial_mlm_score = compute_mlm_score(ngram_sentence, model, tokenizer)
                sequence_mlm_score = initial_mlm_score[0]
                print(f"Sequence MLM score: {sequence_mlm_score}")

                if sequence_mlm_score >= threshold:
                    print(f"Sequence MLM score {sequence_mlm_score} meets the threshold {threshold}. Computing individual word scores...")
                    comparison_matrix = ['*'] * len(ngram_sentence.split())
                    new_pattern = rough_pos.split()
                    words = ngram_sentence.split()
                    rough_pos_tokens = rough_pos.split()

                    if len(words) != len(rough_pos_tokens):
                        print(f"Length mismatch between words and POS tokens for instance ID {instance_id}. Skipping...")
                        continue

                    for i, (pos_tag, word) in enumerate(zip(rough_pos_tokens, words)):
                        word_score = compute_word_scores(word, ngram_sentence, model, tokenizer)
                        print(f"Word '{word}' average score: {word_score}")
                        if word_score >= threshold:
                            new_pattern[i] = word
                            comparison_matrix[i] = word
                        else:
                            new_pattern[i] = pos_tag  # Restore the original POS tag if the word does not meet the score

                    final_hybrid_ngram = ' '.join(new_pattern)

                    if final_hybrid_ngram not in existing_hybrid_ngrams:
                        existing_hybrid_ngrams.add(final_hybrid_ngram)
                        pattern_counter += 1
                        new_pattern_id = generate_pattern_id(pattern_counter)
                        pos_comparison_results.append({
                            'Pattern_ID': new_pattern_id,
                            'POS_N-Gram': rough_pos,
                            'Lexeme_N-Gram': lemma_ngram_sentence,
                            'MLM_Scores': sequence_mlm_score,
                            'Comparison_Replacement_Matrix': ' '.join(comparison_matrix),
                            'Final_Hybrid_N-Gram': final_hybrid_ngram
                        })
                    else:
                        print(f"Hybrid ngram '{final_hybrid_ngram}' already exists. Skipping...")

    # Read existing output data to preserve old patterns
    try:
        with open(output_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            existing_output_data = [row for row in reader]
    except FileNotFoundError:
        existing_output_data = []

    # Write updated results to the output file
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['Pattern_ID', 'POS_N-Gram', 'Lexeme_N-Gram', 'MLM_Scores', 'Comparison_Replacement_Matrix', 'Final_Hybrid_N-Gram']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_output_data + pos_comparison_results)

    print(f"Results saved to {output_file}")

# Example usage
for n in range(2, 8):
    ngram_list_file = 'database/ngrams.csv'
    pos_patterns_file = f'database/Generalized/POSTComparison/{n}grams.csv'
    id_array_file =  f'database/POS/{n}grams.csv'
    output_file = f'database/Generalized/LexemeComparison/{n}grams.csv'

    print(f"Starting generalization for {n}-grams...")
    generalize_patterns(ngram_list_file, pos_patterns_file, id_array_file, output_file, roberta_model, roberta_tokenizer)
    print(f"Finished generalization for {n}-grams.")
