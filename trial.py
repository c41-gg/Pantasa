import nltk
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import torch
import numpy as np

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Define the models
tagalog_bert_model = "jcblaise/bert-tagalog-base-cased"
tagalog_roberta_model = "jcblaise/roberta-tagalog-base"

# Authenticate with Hugging Face if needed (provide your token if required)
# token = "your_huggingface_token"

# Load pre-trained Tagalog BERT and RoBERTa models
bert_tokenizer = AutoTokenizer.from_pretrained(tagalog_bert_model)  # , use_auth_token=token)
bert_model = AutoModelForMaskedLM.from_pretrained(tagalog_bert_model) 

roberta_tokenizer = AutoTokenizer.from_pretrained(tagalog_roberta_model)  # , use_auth_token=token)
roberta_model = AutoModelForMaskedLM.from_pretrained(tagalog_roberta_model)  # , use_auth_token=token)

# Load your dataset
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return data

# Sentence tokenization
def tokenize_sentences(text):
    return sent_tokenize(text)

# POS tagging and lemmatization
def pos_tagging_and_lemmatization(sentences):
    lemmatizer = WordNetLemmatizer()
    pos_tagged_sentences = []
    for sentence in sentences:
        pos_tagged = pos_tag(sentence.split())
        lemmatized = [(lemmatizer.lemmatize(word), tag) for word, tag in pos_tagged]
        pos_tagged_sentences.append(lemmatized)
    return pos_tagged_sentences

# Compute MLM scores
def compute_mlm_score(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return torch.exp(loss).item()

# Compute MPoSM scores (simplified version for illustration)
def compute_mposm_score(sentence, model, tokenizer):
    pos_masked_sentence = ' '.join(['[MASK]' if word != '[MASK]' else word for word in sentence.split()])
    inputs = tokenizer(pos_masked_sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return torch.exp(loss).item()

# Clustering the tagged corpus
def cluster_sentences(sentences, n_clusters=3):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([' '.join([word for word, tag in sentence]) for sentence in sentences])
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans.labels_

# Pattern extraction
def extract_patterns(sentences, mlm_scores, mposm_scores, threshold=0.75):
    patterns = []
    for sentence, mlm_score, mposm_score in zip(sentences, mlm_scores, mposm_scores):
        if mlm_score >= threshold and mposm_score >= threshold:
            patterns.append(sentence)
    return patterns

def extract_pos_patterns(pos_tagged_sentences):
    pos_patterns = [' '.join([tag for word, tag in sentence]) for sentence in pos_tagged_sentences]
    return pos_patterns

# Example process flow
def main():
    dataset = load_dataset("C:/Users/Jarlson/OneDrive/Documents/3rd AY/2nd sem/thesis/dataset.txt")
    tokenized_sentences = tokenize_sentences(' '.join(dataset))
    tagged_sentences = pos_tagging_and_lemmatization(tokenized_sentences)

    mlm_scores = [compute_mlm_score(' '.join([word for word, tag in sentence]), bert_model, bert_tokenizer) for sentence in tagged_sentences]
    mposm_scores = [compute_mposm_score(' '.join([word for word, tag in sentence]), roberta_model, roberta_tokenizer) for sentence in tagged_sentences]

    clusters = cluster_sentences(tagged_sentences, n_clusters=3)
    print("Clusters:", clusters)

    pos_patterns = extract_pos_patterns(tagged_sentences)
    print("POS Tag Patterns:", pos_patterns)

    patterns = extract_patterns(tagged_sentences, mlm_scores, mposm_scores)
    print("Extracted Patterns:", patterns)

if __name__ == "__main__":
    main()
