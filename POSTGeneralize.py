import csv
from collections import Counter
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
    # Flatten the list if it's nested
    if isinstance(mposm_scores[0], list):
        mposm_scores = [score for sublist in mposm_scores for score in sublist]
    return mposm_scores

def compare_pos_sequences(rough_pos, detailed_pos, model, tokenizer, threshold=0.80):
    rough_tokens = rough_pos.split()
    detailed_tokens = detailed_pos.split()
    
    if len(rough_tokens) != len(detailed_tokens):
        print(f"Length mismatch: {len(rough_tokens)} (rough) vs {len(detailed_tokens)} (detailed)")
        return None, None, None, None  # Cannot compare sequences of different lengths

    rough_scores = compute_mposm_scores(' '.join(rough_tokens), model, tokenizer)
    detailed_scores = compute_mposm_scores(' '.join(detailed_tokens), model, tokenizer)

    comparison_matrix = []
    new_pattern = detailed_tokens.copy()  # Start with the detailed pattern

    # Ensure rough_scores and detailed_scores are lists
    if not isinstance(rough_scores, list):
        rough_scores = [rough_scores] * len(rough_tokens)
    if not isinstance(detailed_scores, list):
        detailed_scores = [detailed_scores] * len(detailed_tokens)

    for i in range(len(detailed_tokens)):
        rough_token = rough_tokens[i]
        detailed_token = detailed_tokens[i]
        rough_score = rough_scores[i]
        detailed_score = detailed_scores[i]

        if rough_score != 0 and detailed_score / rough_score >= threshold:
            # Replace "*" with the detailed token in the comparison matrix
            comparison_matrix.append(detailed_token)
        else:
            # Keep "*" indicating no change
            comparison_matrix.append('*')

    # Format comparison matrix as a string
    comparison_matrix_str = ' '.join(comparison_matrix)

    return rough_scores, detailed_scores, comparison_matrix_str, ' '.join(new_pattern)

def generate_pattern_id(ngram_size, counter):
    return f"{ngram_size}{counter:05d}"

def process_pos_patterns(pos_patterns_file, generated_ngrams_file, output_file, model, tokenizer, threshold=0.80):
    pos_patterns = load_csv(pos_patterns_file)
    generated_ngrams = load_csv(generated_ngrams_file)
    
    print(f"Loaded {len(pos_patterns)} POS patterns")
    print(f"Loaded {len(generated_ngrams)} generated n-grams")

    pattern_counter = 1
    pos_comparison_results = []

    for pattern in pos_patterns:
        pattern_id = pattern['Pattern_ID']
        rough_pos = pattern['RoughPOS_N-Gram']
        detailed_pos = pattern['DetailedPOS_N-Gram']
        id_array = pattern['ID_Array'].split(',') if pattern['ID_Array'] else []

        if rough_pos:
            # Process as a rough POS pattern
            instances = [ngram for ngram in generated_ngrams if ngram['N-Gram_ID'] in id_array]
            for instance in instances:
                detailed_pos_instance = instance['DetailedPOS_N-Gram']
                if detailed_pos_instance:
                    rough_scores, detailed_scores, comparison_matrix, new_pattern = compare_pos_sequences(rough_pos, detailed_pos_instance, model, tokenizer, threshold)
                    if rough_scores and detailed_scores:
                        new_pattern_id = generate_pattern_id(len(rough_pos.split()), pattern_counter)
                        pattern_counter += 1
                        # Check if pattern_id already exists in pos_comparison_results
                        if not any(result['Pattern_ID'] == pattern_id for result in pos_comparison_results):
                            pos_comparison_results.append({
                                'Pattern_ID': pattern_id,
                                'RoughPOS_N-Gram': rough_pos,
                                'RPOSN_Freq': None,
                                'DetailedPOS_N-Gram': detailed_pos_instance,
                                'DPOSN_Freq': None,
                                'Comparison_Replacement_Matrix': comparison_matrix,
                                'POS_N-Gram': new_pattern
                            })
                            print(f"Comparison made: Rough POS - {rough_pos}, Detailed POS - {detailed_pos_instance}")
                        else:
                            print(f"Pattern ID {pattern_id} already exists in results, skipping.")
                    else:
                        print(f"Failed comparison: Rough POS - {rough_pos}, Detailed POS - {detailed_pos_instance}")
                else:
                    print(f"No detailed POS instance found for Rough POS - {rough_pos}")

            # Append the original rough POS pattern if it doesn't exist in results
            if not any(result['Pattern_ID'] == pattern_id for result in pos_comparison_results):
                pos_comparison_results.append({
                    'Pattern_ID': pattern_id,
                    'RoughPOS_N-Gram': rough_pos,
                    'RPOSN_Freq': None,
                    'DetailedPOS_N-Gram': None,
                    'DPOSN_Freq': None,
                    'Comparison_Replacement_Matrix': None,
                    'POS_N-Gram': rough_pos
                })
        
        elif detailed_pos:
            # Process as a detailed POS pattern
            overlap_patterns = [p for p in pos_patterns if p['Pattern_ID'] != pattern_id and any(id in id_array for id in p['ID_Array'].split(','))]
            for overlap_pattern in overlap_patterns:
                overlap_rough_pos = overlap_pattern['RoughPOS_N-Gram']
                if overlap_rough_pos:
                    rough_scores, detailed_scores, comparison_matrix, new_pattern = compare_pos_sequences(overlap_rough_pos, detailed_pos, model, tokenizer, threshold)
                    if rough_scores and detailed_scores:
                        new_pattern_id = generate_pattern_id(len(detailed_pos.split()), pattern_counter)
                        pattern_counter += 1
                        # Check if overlap_pattern['Pattern_ID'] already exists in pos_comparison_results
                        if not any(result['Pattern_ID'] == overlap_pattern['Pattern_ID'] for result in pos_comparison_results):
                            pos_comparison_results.append({
                                'Pattern_ID': overlap_pattern['Pattern_ID'],
                                'RoughPOS_N-Gram': overlap_rough_pos,
                                'RPOSN_Freq': None,
                                'DetailedPOS_N-Gram': detailed_pos,
                                'DPOSN_Freq': None,
                                'Comparison_Replacement_Matrix': comparison_matrix,
                                'POS_N-Gram': new_pattern
                            })
                            print(f"Comparison made: Rough POS - {overlap_rough_pos}, Detailed POS - {detailed_pos}")
                        else:
                            print(f"Pattern ID {overlap_pattern['Pattern_ID']} already exists in results, skipping.")
                    else:
                        print(f"Failed comparison: Rough POS - {overlap_rough_pos}, Detailed POS - {detailed_pos}")

            # Append the detailed POS pattern directly to results if not already present
            if not any(result['Pattern_ID'] == pattern_id for result in pos_comparison_results):
                pos_comparison_results.append({
                    'Pattern_ID': pattern_id,
                    'RoughPOS_N-Gram': None,
                    'RPOSN_Freq': None,
                    'DetailedPOS_N-Gram': detailed_pos,
                    'DPOSN_Freq': None,
                    'Comparison_Replacement_Matrix': None,
                    'POS_N-Gram': detailed_pos
                })

    if not pos_comparison_results:
        print("No comparison results were generated.")

    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['Pattern_ID', 'RoughPOS_N-Gram', 'RPOSN_Freq', 'DetailedPOS_N-Gram', 'DPOSN_Freq', 'Comparison_Replacement_Matrix', 'POS_N-Gram']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in pos_comparison_results:
            writer.writerow(result)

# Example usage
for n in range(2, 8):
    ngram_csv = 'database/ngrams.csv'
    pattern_csv = f'database/POS/{n}grams.csv'
    output_csv = f'database/Generalized/POSTComparison/{n}grams.csv'

    process_pos_patterns(pattern_csv, ngram_csv, output_csv, roberta_model, roberta_tokenizer)
