import sys
import os
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.preprocessing.Tokenizer import tokenize
from Modules.preprocessing.POSDTagger import pos_tag as pos_dtag
from Modules.preprocessing.POSRTagger import pos_tag as pos_rtag

def load_dataset(file_path, batch_size=5000):
    """
    Load the dataset in batches of sentences.
    Yields a batch of sentences from the dataset.
    """
    logging.info(f"Loading dataset from {file_path} in batches of {batch_size}")
    with open(file_path, 'r', encoding='utf-8') as file:
        batch = []
        for line in file:
            sentence = line.strip()
            if sentence:
                batch.append(sentence)
            # If the batch size is reached, yield the batch
            if len(batch) == batch_size:
                logging.info(f"Loaded a batch of {len(batch)} sentences.")
                yield batch
                batch = []  # Clear the batch after yielding

        # Yield any remaining sentences in the last batch
        if batch:
            logging.info(f"Loaded the last batch of {len(batch)} sentences.")
            yield batch

def preprocess_text_in_batches(input_file, output_file, batch_size=5000, log_every=500):
    """
    Process the input dataset in batches by tokenizing and performing POS tagging,
    writing the output batch by batch.
    """
    total_tagged_sentences = 0
    log_counter = 0  # To track logging after every 500 sentences
    
    with open(output_file, 'a', encoding='utf-8') as output:  # Changed to 'w' mode for fresh start
        # Load the dataset in batches
        for batch in load_dataset(input_file, batch_size):
            tokenized_sentences = []
            sentence_identifiers = []

            # Process each sentence in the current batch
            for sentence in batch:
                identifier = sentence[:10]  # Use the first 10 characters as an identifier
                sentences_batch = tokenize(sentence)
                tokenized_sentences.extend(sentences_batch)
                sentence_identifiers.extend([identifier] * len(sentences_batch))
            
            # Process the batch once loaded
            process_batch(sentence_identifiers, tokenized_sentences, output)

            # Update total tagged sentences and log after every 500 sentences
            total_tagged_sentences += len(tokenized_sentences)
            while total_tagged_sentences >= log_counter + log_every:
                log_counter += log_every
                logging.info(f"Tagged {log_counter} sentences so far.")

        # Final log for any remaining sentences
        if total_tagged_sentences % log_every != 0:
            logging.info(f"Final batch: tagged {total_tagged_sentences} sentences total.")

    logging.info(f"Preprocessed data saved to {output_file}")
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

    preprocess_text_in_batches(input_file, output_file)  # Apply preprocessing and POS tagging

if __name__ == "__main__":
    main()
