import pandas as pd
from Tokenizer import tokenize
from POSDTagger import pos_tag as pos_dtag
from POSRTagger import pos_tag as pos_rtag
import os
import sys

# Set the JVM options to increase the heap size
os.environ['JVM_OPTS'] = '-Xmx2g'


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df['text'].tolist()

def save_text_file(text_data, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        for line in text_data:
            f.write(line + "\n")

def preprocess_text(input_file, tokenized_file, output_file, batch_size=300):
    dataset = load_dataset(input_file)
    tokenized_sentences = []
    general_pos_tagged_sentences = []
    detailed_pos_tagged_sentences = []

    with open(tokenized_file, 'w', encoding='utf-8') as token_file:
        for text in dataset:
            sentences = tokenize(text)
            token_file.write("\n".join(sentences) + "\n")
            tokenized_sentences.extend(sentences)
        print(f"Sentences tokenized to {tokenized_file}")

    with open(output_file, 'a', encoding='utf-8') as output:
        for i in range(0, len(tokenized_sentences), batch_size):
            batch = tokenized_sentences[i:i + batch_size]

            general_pos_tagged_batch = []
            detailed_pos_tagged_batch = []

            for sentence in batch:
                if sentence:
                    general_pos_tagged_batch.append(pos_rtag(sentence))
                    detailed_pos_tagged_batch.append(pos_dtag(sentence))
                else:
                    general_pos_tagged_batch.append('')
                    detailed_pos_tagged_batch.append('')

            # Append batch results to output file immediately
            for tok_sentence, gen_pos, det_pos in zip(batch, general_pos_tagged_batch, detailed_pos_tagged_batch):
                output.write(f"{tok_sentence},{gen_pos},{det_pos},\n")

            # Clear lists after each batch to avoid memory issues
            tokenized_sentences.clear()
            general_pos_tagged_sentences.clear()
            detailed_pos_tagged_sentences.clear()

    print(f"Preprocessed data saved to {output_file}")

def main(input_csv, tokenized_txt, output_csv):
    preprocess_text(input_csv, tokenized_txt, output_csv)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python Preprocessing.py <input_csv> <tokenized_txt> <output_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    tokenized_txt = sys.argv[2]
    output_csv = sys.argv[3]

    main(input_csv, tokenized_txt, output_csv)
