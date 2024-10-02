import sys
import os
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.preprocessing.Tokenizer import tokenize
from Modules.preprocessing.POSDTagger import pos_tag as pos_dtag
from Modules.preprocessing.POSRTagger import pos_tag as pos_rtag

def load_dataset(file_path):
    logging.info(f"Loading dataset from {file_path}")
    # Read the file as a plain text file, one sentence per line
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # Strip leading/trailing spaces from each line and return as a list
    sentences = [line.strip() for line in lines if line.strip()]
    logging.info(f"Loaded {len(sentences)} sentences from the dataset.")
    return sentences

def preprocess_text(input_file, output_file, batch_size=5000, log_every=50):
    """
    Process the input dataset by tokenizing and performing POS tagging, 
    writing the output in batches to avoid memory overload.
    """
    with open(output_file, 'a', encoding='utf-8') as output:
        tokenized_sentences = []
        sentence_identifiers = []
        
        # Load the dataset by passing the input_file to load_dataset
        sentences = load_dataset(input_file)
        total_sentences = len(sentences)
        
        total_tagged_sentences = 0
        log_counter = 0  # To track the number of sentences tagged for logging
        
        # Stream the dataset
        for sentence in sentences:
            identifier = sentence[:10]  # Example of a basic identifier (first 10 characters)
            sentences_batch = tokenize(sentence)
            tokenized_sentences.extend(sentences_batch)
            sentence_identifiers.extend([identifier] * len(sentences_batch))
            
            # Process in batches
            if len(tokenized_sentences) >= batch_size:
                process_batch(sentence_identifiers, tokenized_sentences, output)
                total_tagged_sentences += len(tokenized_sentences)
                tokenized_sentences = []  # Clear the list after processing
                sentence_identifiers = []  # Clear the identifier list after processing
            
            # Log every time 500 sentences are tagged
            if total_tagged_sentences >= log_counter + log_every:
                log_counter += log_every
                logging.info(f"Tagged {log_counter} sentences so far.")

        # If there are any remaining sentences in the last incomplete batch
        if tokenized_sentences:
            process_batch(sentence_identifiers, tokenized_sentences, output)
            total_tagged_sentences += len(tokenized_sentences)
    
    logging.info(f"Preprocessed data saved to {output_file}")
    logging.info(f"Total sentences processed: {total_sentences}")
    logging.info(f"Total sentences tagged and saved: {total_tagged_sentences}")

def process_batch(sentence_identifiers, tokenized_sentences, output_file):
    """
    Helper function to process a batch of tokenized sentences and write to the output file.
    Each sentence is written along with its identifier.
    """
    general_pos_tagged_batch = []
    detailed_pos_tagged_batch = []
    
    # Perform POS tagging for each sentence
    for sentence in tokenized_sentences:
        if sentence:
            general_pos_tagged_batch.append(pos_rtag(sentence))  # Rough POS tagging
            detailed_pos_tagged_batch.append(pos_dtag(sentence))  # Detailed POS tagging
        else:
            general_pos_tagged_batch.append('')
            detailed_pos_tagged_batch.append('')

    # Write tokenized sentences and their POS tags to the output file
    for identifier, tok_sentence, gen_pos, det_pos in zip(sentence_identifiers, tokenized_sentences, general_pos_tagged_batch, detailed_pos_tagged_batch):
        # Add quotes around sentences that contain commas to handle CSV format properly
        if ',' in tok_sentence:
            tok_sentence = f'"{tok_sentence}"'
        output_file.write(f"{identifier},{tok_sentence},{gen_pos},{det_pos}\n")

    logging.info(f"Processed and saved {len(tokenized_sentences)} sentences in the batch.")

    # Clear lists after processing to save memory
    general_pos_tagged_batch.clear()
    detailed_pos_tagged_batch.clear()

# Main function to execute the preprocessing
def main():
    input_file = 'dataset/ALT-Parallel-Corpus-20191206/data_fil.txt'   # Input file with raw text
    output_file = 'MPoSM/preprocessed_output.csv'  # Output file to save processed data

    preprocess_text(input_file, output_file)  # Apply preprocessing and POS tagging

if __name__ == "__main__":
    main()
