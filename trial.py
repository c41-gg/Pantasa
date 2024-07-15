from Tokenizer import tokenize
from POSDTagger import pos_tag as pos_dtag
from POSRTagger import pos_tag as pos_rtag
from GramSizeClustering import cluster_ngrams
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from nltk.stem import WordNetLemmatizer

# Define the models
tagalog_bert_model = "jcblaise/bert-tagalog-base-cased"
tagalog_roberta_model = "jcblaise/roberta-tagalog-base"

# Load pre-trained Tagalog BERT and RoBERTa models
bert_tokenizer = AutoTokenizer.from_pretrained(tagalog_bert_model)
bert_model = AutoModelForMaskedLM.from_pretrained(tagalog_bert_model)

roberta_tokenizer = AutoTokenizer.from_pretrained(tagalog_roberta_model)
roberta_model = AutoModelForMaskedLM.from_pretrained(tagalog_roberta_model)

lemmatizer = WordNetLemmatizer()

def load_dataset(file_path):
  with open(file_path, 'r', encoding='utf-8') as file:
    data = [line.strip().split(',') for line in file.readlines()]  # Split by comma
  return data

def lemmatize_sentence(sentence):
    words = sentence.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def compute_mlm_and_mposm_scores(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    mlm_loss = outputs.loss
    mlm_scores = torch.exp(outputs.logits).squeeze().tolist()  # MLM scores for each token

    pos_masked_sentence = ' '.join(['[MASK]' if word != '[MASK]' else word for word in sentence.split()])
    inputs = tokenizer(pos_masked_sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    mposm_loss = outputs.loss
    mposm_scores = torch.exp(outputs.logits).squeeze().tolist()  # MPOSM scores for each token

    return mlm_scores, mposm_scores

def extract_patterns(tagged_sentences, lemmatized_sentences, mlm_scores_bert, mposm_scores_bert, mlm_scores_roberta, mposm_scores_roberta, threshold=0.75):
    patterns = []
    for tagged_sentence, lemmatized_sentence, mlm_score_bert, mposm_score_bert, mlm_score_roberta, mposm_score_roberta in zip(tagged_sentences, lemmatized_sentences, mlm_scores_bert, mposm_scores_bert, mlm_scores_roberta, mposm_scores_roberta):
        words = lemmatized_sentence.split()
        tags = tagged_sentence.split()
        new_pattern = []
        for i, (word, tag, mlm_b, mposm_b, mlm_r, mposm_r) in enumerate(zip(words, tags, mlm_score_bert, mposm_score_bert, mlm_score_roberta, mposm_score_roberta)):
            if (mlm_b >= threshold or mlm_r >= threshold):
                new_pattern.append(word)
            else:
                new_pattern.append(tag)
        patterns.append(' '.join(new_pattern))
    return patterns

def main():
    dataset = load_dataset("database/test-dataset.csv")
    text = ' '.join(dataset)
    
    sentences = tokenize(text)
    print("\nTOKENIZED VERSION!\n") 
    for sentence in sentences:
        print(sentence)
    
    print("\nGENERAL POS TAGGED VERSION!\n")
    general_pos_tagged_sentences = []
    for sentence in sentences:
        tagged_sentence = pos_rtag(sentence)
        general_pos_tagged_sentences.append(tagged_sentence)
        print(tagged_sentence)

    print("\nDETAILED POS TAGGED VERSION!\n")
    detailed_pos_tagged_sentences = []
    for sentence in sentences:
        tagged_sentence = pos_dtag(sentence)
        detailed_pos_tagged_sentences.append(tagged_sentence)
        print(tagged_sentence)

    print("\nLEMMATIZED SENTENCES!\n")
    lemmatized_sentences = [lemmatize_sentence(sentence) for sentence in detailed_pos_tagged_sentences]
    for sentence in lemmatized_sentences:
        print(sentence)

    # Clustering
    print("\nCLUSTERING BASED ON N-GRAMS!\n")
    clustered_ngrams = cluster_ngrams(lemmatized_sentences)
    for n in sorted(clustered_ngrams):
        print(f"\nClustered {n}-grams:\n")
        for ngram in clustered_ngrams[n]:
            print(ngram)

    # Compute scores
    mlm_scores_bert, mposm_scores_bert = zip(*[compute_mlm_and_mposm_scores(sentence, bert_model, bert_tokenizer) for sentence in lemmatized_sentences])
    mlm_scores_roberta, mposm_scores_roberta = zip(*[compute_mlm_and_mposm_scores(sentence, roberta_model, roberta_tokenizer) for sentence in lemmatized_sentences])

    # Extract patterns
    patterns = extract_patterns(detailed_pos_tagged_sentences, lemmatized_sentences, mlm_scores_bert, mposm_scores_bert, mlm_scores_roberta, mposm_scores_roberta)
    print("\nEXTRACTED PATTERNS!\n")
    for pattern in patterns:
        print(pattern)

if __name__ == "__main__":
    main()