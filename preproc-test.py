import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from Tokenizer import tokenize
from POSDTagger import pos_tag as pos_dtag
from POSRTagger import pos_tag  as pos_rtag

# Make sure to download the necessary NLTK data files if not already done
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load your dataset
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return data

# Sentence tokenization
def tokenize_sentences(text):
    return tokenize(text)

# Function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# POS tagging and lemmatization
def pos_tagging_and_lemmatization(sentences):
    lemmatizer = WordNetLemmatizer()
    processed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        pos_dtagged = pos_dtag(sentence)
        pos_rtagged = pos_rtag(sentence)  # Rough POS tags (for this example, they are same as detailed tags)
        lemmatized = [(lemmatizer.lemmatize(words, get_wordnet_pos(tag)), tag) for words, tag in pos_dtagged]
        
        processed_sentences.append({
            'sentence': sentence,
            'detailed_pos_tags': ' '.join([tag for _, tag in pos_dtagged]),
            'rough_pos_tags': ' '.join(pos_rtagged),
            'lemmatized_sentence': ' '.join([word for word, _ in lemmatized])
        })
    return processed_sentences

# Save processed data to CSV
def save_to_csv(data, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['sentence', 'detailed_pos_tags', 'rough_pos_tags', 'lemmatized_sentence'])
        writer.writeheader()
        for row in data:
            writer.writerow(row)

if __name__ == "__main__":
    # Load data from the dataset CSV file
    input_csv = "database/dataset.csv"
    data = load_dataset(input_csv)
    
    # Process each paragraph and tokenize into sentences
    processed_data = []
    for paragraph in data:
        sentences = tokenize_sentences(paragraph.strip())
        processed_sentences = pos_tagging_and_lemmatization(sentences)
        processed_data.extend(processed_sentences)
    
    # Save the processed data to the preprocessed CSV file
    output_csv = "database/preprocessed.csv"
    save_to_csv(processed_data, output_csv)
