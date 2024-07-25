import csv
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Define the models
tagalog_roberta_model = "jcblaise/roberta-tagalog-base"
roberta_tokenizer = AutoTokenizer.from_pretrained(tagalog_roberta_model)
roberta_model = AutoModelForMaskedLM.from_pretrained(tagalog_roberta_model)

def load_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data

def compute_mposm_scores(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    mposm_scores = torch.exp(outputs.logits).squeeze().tolist()  # MPoSM scores for each token
    if isinstance(mposm_scores[0], list):
        mposm_scores = [score for sublist in mposm_scores for score in sublist]
    return mposm_scores

def compare_pos_sequences(rough_pos, detailed_pos, model, tokenizer, threshold=0.80):
    rough_tokens = rough_pos.split()
    detailed_tokens = detailed_pos.split()
    
    if len(rough_tokens) != len(detailed_tokens):
        print(f"Length mismatch: {len(rough_tokens)} (rough) vs {len(detailed_tokens)} (detailed)")
        return None, None, None, None

    rough_scores = compute_mposm_scores(' '.join(rough_tokens), model, tokenizer)
    detailed_scores = compute_mposm_scores(' '.join(detailed_tokens), model, tokenizer)

    comparison_matrix = []
    for i in range(len(detailed_tokens)):
        rough_token = rough_tokens[i]
        detailed_token = detailed_tokens[i]
        rough_score = rough_scores[i]
        detailed_score = detailed_scores[i]

        if rough_score != 0 and detailed_score / rough_score >= threshold:
            comparison_matrix.append(detailed_token)
        else:
            comparison_matrix.append('*')

    new_pattern = [comparison_matrix[i] if comparison_matrix[i] != '*' else rough_tokens[i] for i in range(len(rough_tokens))]
    comparison_matrix_str = ' '.join(comparison_matrix)
    new_pattern_str = ' '.join(new_pattern)

    return rough_scores, detailed_scores, comparison_matrix_str, new_pattern_str

def generate_pattern_id(counter):
    return f"{counter:06d}"

def collect_existing_patterns(file_path):
    patterns = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['POS_N-Gram']:
                    patterns.add(row['POS_N-Gram'])
    except FileNotFoundError:
        pass
    return patterns

def get_latest_pattern_id(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            pattern_ids = [int(row['Pattern_ID']) for row in reader if row['Pattern_ID'].isdigit()]
            return max(pattern_ids, default=0)
    except FileNotFoundError:
        return 0

def process_pos_patterns(pos_patterns_file, generated_ngrams_file, pattern_file, output_file, model, tokenizer, threshold=0.80):
    pos_patterns = load_csv(pos_patterns_file)
    generated_ngrams = load_csv(generated_ngrams_file)

    print(f"Loaded {len(pos_patterns)} POS patterns")
    print(f"Loaded {len(generated_ngrams)} generated n-grams")

    existing_patterns_input = set()
    existing_patterns_output = collect_existing_patterns(output_file)

    # Collect existing patterns from input CSV
    for pattern in pos_patterns:
        if pattern['RoughPOS_N-Gram']:
            existing_patterns_input.add(pattern['RoughPOS_N-Gram'])
        if pattern['DetailedPOS_N-Gram']:
            existing_patterns_input.add(pattern['DetailedPOS_N-Gram'])

    latest_pattern_id_input = get_latest_pattern_id(pos_patterns_file)
    latest_pattern_id_output = get_latest_pattern_id(output_file)
    latest_pattern_id = max(latest_pattern_id_input, latest_pattern_id_output)
    pattern_counter = latest_pattern_id + 1

    pos_comparison_results = []
    new_patterns = []
    seen_patterns = {}

    for pattern in pos_patterns:
        pattern_id = pattern['Pattern_ID']
        rough_pos = pattern['RoughPOS_N-Gram']
        detailed_pos = pattern['DetailedPOS_N-Gram']
        id_array = pattern['ID_Array'].split(',') if pattern['ID_Array'] else []

        if rough_pos and rough_pos in existing_patterns_input:
            instances = [ngram for ngram in generated_ngrams if ngram['N-Gram_ID'] in id_array]
            for instance in instances:
                detailed_pos_instance = instance['DetailedPOS_N-Gram']
                if detailed_pos_instance:
                    rough_scores, detailed_scores, comparison_matrix, new_pattern = compare_pos_sequences(rough_pos, detailed_pos_instance, model, tokenizer, threshold)
                    if rough_scores and detailed_scores:
                        if new_pattern not in existing_patterns_input and new_pattern not in existing_patterns_output:
                            new_pattern_id = generate_pattern_id(pattern_counter)
                            seen_patterns[new_pattern] = new_pattern_id
                            pattern_counter += 1
                            pos_comparison_results.append({
                                'Pattern_ID': new_pattern_id,
                                'RoughPOS_N-Gram': rough_pos,
                                'RPOSN_Freq': None,
                                'DetailedPOS_N-Gram': detailed_pos_instance,
                                'DPOSN_Freq': None,
                                'Comparison_Replacement_Matrix': comparison_matrix,
                                'POS_N-Gram': new_pattern
                            })
                            new_patterns.append({
                                'Pattern_ID': new_pattern_id,
                                'RoughPOS_N-Gram': rough_pos,
                                'DetailedPOS_N-Gram': detailed_pos_instance,
                                'Frequency': len(id_array),
                                'ID_Array': ','.join(id_array)
                            })
                            print(f"Comparison made: Rough POS - {rough_pos}, Detailed POS - {detailed_pos_instance}")
                else:
                    print(f"No detailed POS instance found for Rough POS - {rough_pos}")

            if rough_pos not in seen_patterns and rough_pos not in existing_patterns_output:
                pos_comparison_results.append({
                    'Pattern_ID': pattern_id,
                    'RoughPOS_N-Gram': rough_pos,
                    'RPOSN_Freq': None,
                    'DetailedPOS_N-Gram': None,
                    'DPOSN_Freq': None,
                    'Comparison_Replacement_Matrix': None,
                    'POS_N-Gram': rough_pos
                })
                seen_patterns[rough_pos] = pattern_id
        
        elif detailed_pos and detailed_pos in existing_patterns_input:
            overlap_patterns = [p for p in pos_patterns if p['Pattern_ID'] != pattern_id and any(id in id_array for id in p['ID_Array'].split(','))]
            for overlap_pattern in overlap_patterns:
                overlap_rough_pos = overlap_pattern['RoughPOS_N-Gram']
                if overlap_rough_pos:
                    rough_scores, detailed_scores, comparison_matrix, new_pattern = compare_pos_sequences(overlap_rough_pos, detailed_pos, model, tokenizer, threshold)
                    if rough_scores and detailed_scores:
                        if new_pattern not in existing_patterns_input and new_pattern not in existing_patterns_output:
                            new_pattern_id = generate_pattern_id(pattern_counter)
                            seen_patterns[new_pattern] = new_pattern_id
                            pattern_counter += 1
                            pos_comparison_results.append({
                                'Pattern_ID': new_pattern_id,
                                'RoughPOS_N-Gram': overlap_rough_pos,
                                'RPOSN_Freq': None,
                                'DetailedPOS_N-Gram': detailed_pos,
                                'DPOSN_Freq': None,
                                'Comparison_Replacement_Matrix': comparison_matrix,
                                'POS_N-Gram': new_pattern
                            })
                            new_patterns.append({
                                'Pattern_ID': new_pattern_id,
                                'RoughPOS_N-Gram': overlap_rough_pos,
                                'DetailedPOS_N-Gram': detailed_pos,
                                'Frequency': len(id_array),
                                'ID_Array': ','.join(id_array)
                            })
                            print(f"Comparison made: Rough POS - {overlap_rough_pos}, Detailed POS - {detailed_pos}")
                else:
                    print(f"No overlap rough POS instance found for Detailed POS - {detailed_pos}")

            if detailed_pos not in seen_patterns and detailed_pos not in existing_patterns_output:
                pos_comparison_results.append({
                    'Pattern_ID': pattern_id,
                    'RoughPOS_N-Gram': None,
                    'RPOSN_Freq': None,
                    'DetailedPOS_N-Gram': detailed_pos,
                    'DPOSN_Freq': None,
                    'Comparison_Replacement_Matrix': None,
                    'POS_N-Gram': detailed_pos
                })
                seen_patterns[detailed_pos] = pattern_id

    if not pos_comparison_results:
        print("No comparison results were generated.")

    # Read existing data to append new results
    existing_data = []
    try:
        with open(output_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            existing_data = [row for row in reader]
    except FileNotFoundError:
        pass  # No existing data

    # Write all results (existing + new)
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['Pattern_ID', 'RoughPOS_N-Gram', 'RPOSN_Freq', 'DetailedPOS_N-Gram', 'DPOSN_Freq', 'Comparison_Replacement_Matrix', 'POS_N-Gram']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader(\
           \ 
        for result in existing_data + pos_comparison_results:
            writer.writerow(result)

    # Append new patterns to the pattern CSV
    with open(pattern_file, 'a', newline='', encoding='utf-8') as file:
        fieldnames = ['Pattern_ID', 'RoughPOS_N-Gram', 'DetailedPOS_N-Gram', 'Frequency', 'ID_Array']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        for new_pattern in new_patterns:
            writer.writerow(new_pattern)

# Example usage
for n in range(2, 8):
    ngram_csv = 'database/ngrams.csv'
    pattern_csv = f'database/POS/{n}grams.csv'
    output_csv = f'database/Generalized/POSTComparison/{n}grams.csv'

    print(f"Processing n-gram size: {n}")  # Debug statement

    process_pos_patterns(pattern_csv, ngram_csv, pattern_csv, output_csv, roberta_model, roberta_tokenizer)
