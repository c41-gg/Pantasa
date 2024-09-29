import re
import csv
import os
import pandas as pd
from Tokenizer import tokenize
from POSDTagger import pos_tag as pos_dtag
from POSRTagger import pos_tag as pos_rtag

def load_dataset(file_path):
    # Read the file as a plain text file, one sentence per line
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # Strip leading/trailing spaces from each line and return as a list
    return [line.strip() for line in lines if line.strip()]

def preprocess_text(input_file, output_file, batch_size=5000):
    # Load the dataset
    dataset = load_dataset(input_file)
    
    # Initialize lists to store tokenized sentences and POS tags
    tokenized_sentences = []
    
    # Tokenize the dataset
    for text in dataset:
        sentences = tokenize(text)
        tokenized_sentences.extend(sentences)
    
    # Process in batches to avoid memory overload
    with open(output_file, 'a', encoding='utf-8') as output:
        for i in range(0, len(tokenized_sentences), batch_size):
            batch = tokenized_sentences[i:i + batch_size]

            general_pos_tagged_batch = []
            detailed_pos_tagged_batch = []

            for sentence in batch:
                if sentence:
                    general_pos_tagged_batch.append(pos_rtag(sentence))  # Rough POS tagging
                    detailed_pos_tagged_batch.append(pos_dtag(sentence))  # Detailed POS tagging
                else:
                    general_pos_tagged_batch.append('')
                    detailed_pos_tagged_batch.append('')

            # Write the tokenized sentences and their POS tags to the output file
            for tok_sentence, gen_pos, det_pos in zip(batch, general_pos_tagged_batch, detailed_pos_tagged_batch):
                # Add quotes around sentences that contain commas
                if ',' in tok_sentence:
                    tok_sentence = f'"{tok_sentence}"'
                output.write(f"{tok_sentence},{gen_pos},{det_pos}\n")

            # Clear batch-specific lists to save memory (except tokenized_sentences, which is used for multiple batches)
            general_pos_tagged_batch.clear()
            detailed_pos_tagged_batch.clear()

    print(f"Preprocessed data saved to {output_file}")

# Main function to execute the preprocessing
def main():
    input_file = 'MPoSM/newsph.txt'   # Input file with raw text
    output_file = 'MPoSM/preprocessed_output.csv'  # Output file to save processed data

    preprocess_text(input_file, output_file)  # Apply preprocessing and POS tagging

if __name__ == "__main__":
    main()
