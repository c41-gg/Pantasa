import csv

def load_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data

def compare_pos_sequences(rough_pos, detailed_pos, rough_freq, detailed_freq, threshold=0.80):
    rough_tokens = rough_pos.split()
    detailed_tokens = detailed_pos.split()
    
    if len(rough_tokens) != len(detailed_tokens):
        print(f"Length mismatch: {len(rough_tokens)} (rough) vs {len(detailed_tokens)} (detailed)")
        return None, None  # Cannot compare sequences of different lengths

    comparison_matrix = []
    new_pattern = []

    for i in range(len(rough_tokens)):
        rough_token = rough_tokens[i]
        detailed_token = detailed_tokens[i]

        if detailed_freq / rough_freq >= threshold:
            new_pattern.append(detailed_token)
            comparison_matrix.append(detailed_token)
        else:
            new_pattern.append(rough_token)
            comparison_matrix.append('*')

    comparison_matrix_str = ' '.join(comparison_matrix)
    new_pattern_str = ' '.join(new_pattern)

    return comparison_matrix_str, new_pattern_str

def process_pos_patterns(pos_patterns_file, generated_ngrams_file, output_file, threshold=0.80):
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
        frequency = int(pattern['Frequency'])
        id_array = pattern['ID_Array'].split(',') if pattern['ID_Array'] else []

        if rough_pos and id_array:
            # Process as a rough POS pattern
            for ngram_id in id_array:
                instance = next((ngram for ngram in generated_ngrams if ngram['N-Gram_ID'] == ngram_id), None)
                if instance:
                    detailed_pos_instance = instance['DetailedPOS_N-Gram']
                    detailed_freq = int(pattern['Frequency'])
                    if detailed_pos_instance:
                        comparison_matrix, new_pattern = compare_pos_sequences(rough_pos, detailed_pos_instance, frequency, detailed_freq, threshold)
                        pos_comparison_results.append({
                            'Pattern_ID': pattern_id,
                            'RoughPOS_N-Gram': rough_pos,
                            'RPOSN_Freq': frequency,
                            'DetailedPOS_N-Gram': detailed_pos_instance,
                            'DPOSN_Freq': detailed_freq,
                            'Comparison_Replacement_Matrix': comparison_matrix,
                            'POS_N-Gram': new_pattern
                        })
                        print(f"Comparison made: Rough POS - {rough_pos}, Detailed POS - {detailed_pos_instance}")
                    else:
                        print(f"No detailed POS instance found for Rough POS - {rough_pos}")

            # Append the original rough POS pattern if it doesn't exist in results
            if not any(result['Pattern_ID'] == pattern_id for result in pos_comparison_results):
                pos_comparison_results.append({
                    'Pattern_ID': pattern_id,
                    'RoughPOS_N-Gram': rough_pos,
                    'RPOSN_Freq': frequency,
                    'DetailedPOS_N-Gram': None,
                    'DPOSN_Freq': None,
                    'Comparison_Replacement_Matrix': None,
                    'POS_N-Gram': rough_pos
                })
        
        elif detailed_pos and id_array:
            # Process as a detailed POS pattern
            for ngram_id in id_array:
                instance = next((ngram for ngram in generated_ngrams if ngram['N-Gram_ID'] == ngram_id), None)
                if instance:
                    overlap_rough_pos = instance['RoughPOS_N-Gram']
                    rough_freq = int(pattern['Frequency'])
                    if overlap_rough_pos:
                        comparison_matrix, new_pattern = compare_pos_sequences(overlap_rough_pos, detailed_pos, rough_freq, frequency, threshold)
                        pos_comparison_results.append({
                            'Pattern_ID': pattern_id,
                            'RoughPOS_N-Gram': overlap_rough_pos,
                            'RPOSN_Freq': rough_freq,
                            'DetailedPOS_N-Gram': detailed_pos,
                            'DPOSN_Freq': frequency,
                            'Comparison_Replacement_Matrix': comparison_matrix,
                            'POS_N-Gram': new_pattern
                        })
                        print(f"Comparison made: Rough POS - {overlap_rough_pos}, Detailed POS - {detailed_pos}")

            # Append the detailed POS pattern directly to results
            pos_comparison_results.append({
                'Pattern_ID': pattern_id,
                'RoughPOS_N-Gram': None,
                'RPOSN_Freq': None,
                'DetailedPOS_N-Gram': detailed_pos,
                'DPOSN_Freq': frequency,
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
    output_csv = f'database/Generalized/RoBERTa/POSTComparison/{n}grams.csv'

    process_pos_patterns(pattern_csv, ngram_csv, output_csv)
