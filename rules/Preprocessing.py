import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Import tqdm for progress tracking
from Modules.Tokenizer import tokenize
from Modules.POSRTagger import pos_tag  # Use the pos_tag function from POSRTagger for both Rough and Detailed POS
from Modules.Lemmatizer import lemmatize_sentence

import sys

# Add the path to morphinas_project
sys.path.append('C:/Users/Carlo Agas/Documents/GitHub/Pantasaa/morphinas_project')

from lemmatizer_client import initialize_stemmer

# Initialize the Morphinas lemmatizer once to reuse across function calls
gateway, lemmatizer = initialize_stemmer()

# Set the JVM options to increase the heap size
os.environ['JVM_OPTS'] = '-Xmx2g'

def load_dataset(file_path):
    """Load dataset from a text file, assuming each line contains a single sentence."""
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentence = line.strip()
            if sentence:
                dataset.append(sentence)
    return dataset

def save_text_file(text_data, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        for line in text_data:
            f.write(line + "\n")

def load_tokenized_sentences(tokenized_file):
    """Load already tokenized sentences from the tokenized file."""
    tokenized_sentences = set()
    if os.path.exists(tokenized_file):
        with open(tokenized_file, 'r', encoding='utf-8') as file:
            for line in file:
                tokenized_sentences.add(line.strip())
    return tokenized_sentences

def load_processed_sentences(output_file):
    """Load already processed sentences from the output file."""
    processed_sentences = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.split(',')  # Assuming the sentence is the first element in each line
                if parts:
                    processed_sentences.add(parts[0].strip('"'))
    return processed_sentences

def process_sentence(sentence):
    """Process a single sentence for general POS tagging, detailed POS tagging, and lemmatization."""
    detailed_pos, general_pos = pos_tag(sentence)  # pos_tag returns both detailed and rough tags
    lemmatized_sentence = lemmatize_sentence(sentence)
    return general_pos, detailed_pos, lemmatized_sentence

def preprocess_text(input_file, tokenized_file, output_file):
    # Dynamically get the maximum number of CPU cores available
    max_workers = os.cpu_count()
    print(f"Number of cores being used: {max_workers}")

    dataset = load_dataset(input_file)
    tokenized_sentences = load_tokenized_sentences(tokenized_file)
    processed_sentences = load_processed_sentences(output_file)

    new_tokenized_sentences = []

    with open(tokenized_file, 'a', encoding='utf-8') as token_file:
        for sentence in dataset:
            # Tokenize the sentence before processing it further
            tokenized_sentence_parts = tokenize(sentence)

            for tokenized_sentence in tokenized_sentence_parts:
                if tokenized_sentence not in tokenized_sentences and tokenized_sentence not in processed_sentences:
                    new_tokenized_sentences.append(tokenized_sentence)
                    tokenized_sentences.add(tokenized_sentence)
                    token_file.write(tokenized_sentence + "\n")

        print(f"Sentences tokenized to {tokenized_file}")

    # Use tqdm on the actual sentences processed
    with open(output_file, 'a', encoding='utf-8') as output:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to show progress for each sentence in new_tokenized_sentences
            for sentence, result in zip(new_tokenized_sentences, tqdm(executor.map(process_sentence, new_tokenized_sentences), total=len(new_tokenized_sentences), desc="Processing Sentences")):
                general_pos, detailed_pos, lemma = result

                # Wrap lemmatized sentence in quotes if it contains a comma
                if ',' in lemma:
                    lemma = f'"{lemma}"'
                
                # Wrap tokenized sentence in quotes if it contains a comma
                if ',' in sentence:
                    sentence = f'"{sentence}"'
                
                output.write(f"{sentence},{general_pos},{detailed_pos},{lemma},\n")

    print(f"Preprocessed data saved to {output_file}")

def run_preprocessing():
    # Define your file paths here
    input_txt = "/content/Pantasa/rules/dataset/ALT-Parallel-Corpus-20191206/data_fil.txt"           # Input file (the .txt file)
    tokenized_txt = "/content/Pantasa/rules/database/tokenized_sentences.txt"  # File to save tokenized sentences
    output_csv = "/content/Pantasa/rules/database/preprocessed.csv"     # File to save the preprocessed output

    # Start the preprocessing
    preprocess_text(input_txt, tokenized_txt, output_csv)

# Automatically run when the script is executed
if __name__ == "__main__":
    run_preprocessing()
