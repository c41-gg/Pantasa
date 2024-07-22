import csv
from collections import defaultdict

def generate_pattern_id(ngram_size, counter):
    return f"{ngram_size}{counter:05d}"

def get_max_pattern_id(output_file):
    max_pattern_id = 0
    try:
        with open(output_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                pattern_id = int(row['Pattern_ID'][1:])  # Skipping the first character
                if pattern_id > max_pattern_id:
                    max_pattern_id = pattern_id
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error reading max pattern ID: {e}")
    return max_pattern_id

def process_clustered_ngrams(input_file, output_file, ngram_size):
    rough_pos_patterns = defaultdict(list)
    detailed_pos_patterns = defaultdict(list)
    max_pattern_id = get_max_pattern_id(output_file)
    
    # Read the clustered n-grams from the input file
    try:
        with open(input_file, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            
            for row in reader:
                ngram_id = row['N-Gram_ID']
                rough_pos = row['RoughPOS_N-Gram']
                detailed_pos = row['DetailedPOS_N-Gram']
                
                if rough_pos:
                    rough_pos_patterns[rough_pos].append((ngram_id, detailed_pos))
                if detailed_pos:
                    detailed_pos_patterns[detailed_pos].append((ngram_id, rough_pos))
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Count the frequency of each pattern
    rough_pos_frequency = {key: len(ids) for key, ids in rough_pos_patterns.items() if len(ids) > 1}
    detailed_pos_frequency = {key: len(ids) for key, ids in detailed_pos_patterns.items() if len(ids) > 1}
    
    # Write the results to the output CSV file
    try:
        with open(output_file, 'a', newline='', encoding='utf-8') as out_file:
            fieldnames = ['Pattern_ID', 'RoughPOS_N-Gram', 'DetailedPOS_N-Gram', 'Frequency', 'ID_Array']
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            
            if max_pattern_id == 0:
                writer.writeheader()
            
            # Write rough POS patterns
            for rough_pos, ids in rough_pos_patterns.items():
                if rough_pos in rough_pos_frequency:
                    pattern_id = generate_pattern_id(ngram_size, max_pattern_id + 1)
                    max_pattern_id += 1
                    writer.writerow({
                        'Pattern_ID': pattern_id,
                        'RoughPOS_N-Gram': rough_pos,
                        'DetailedPOS_N-Gram': '',
                        'Frequency': rough_pos_frequency[rough_pos],
                        'ID_Array': [id[0] for id in ids]  # Array of all IDs
                    })
            
            # Write detailed POS patterns
            for detailed_pos, ids in detailed_pos_patterns.items():
                if detailed_pos in detailed_pos_frequency:
                    pattern_id = generate_pattern_id(ngram_size, max_pattern_id + 1)
                    max_pattern_id += 1
                    writer.writerow({
                        'Pattern_ID': pattern_id,
                        'RoughPOS_N-Gram': '',
                        'DetailedPOS_N-Gram': detailed_pos,
                        'Frequency': detailed_pos_frequency[detailed_pos],
                        'ID_Array': [id[0] for id in ids]  # Array of all IDs
                    })
    except Exception as e:
        print(f"Error writing output file: {e}")

# Example usage
for n in range(2, 8):
    input_csv = f'database/GramSize/{n}grams.csv'
    output_csv = f'database/POS/{n}grams.csv'
    process_clustered_ngrams(input_csv, output_csv, n)
