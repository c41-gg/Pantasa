import pandas as pd
from POSDTagger import pos_tag as pos_dtag
from POSRTagger import pos_tag as pos_rtag
from GramSizeClustering import cluster_ngrams
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch


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
        data = file.readlines()
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
    mlm_score = torch.exp(mlm_loss).item()

    pos_masked_sentence = ' '.join(['[MASK]' if word != '[MASK]' else word for word in sentence.split()])
    inputs = tokenizer(pos_masked_sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    mposm_loss = outputs.loss
    mposm_score = torch.exp(mposm_loss).item()

    return mlm_score, mposm_score

def extract_patterns(tagged_sentences, mlm_scores_bert, mposm_scores_bert, mlm_scores_roberta, mposm_scores_roberta, threshold=0.75):
    patterns = []
    for sentence, mlm_score_bert, mposm_score_bert, mlm_score_roberta, mposm_score_roberta in zip(tagged_sentences, mlm_scores_bert, mposm_scores_bert, mlm_scores_roberta, mposm_scores_roberta):
        if (mlm_score_bert >= threshold and mposm_score_bert >= threshold) or (mlm_score_roberta >= threshold and mposm_score_roberta >= threshold):
            patterns.append(sentence)
    return patterns

def main():
    dataset = load_dataset("C:/Users/Jarlson/OneDrive/Documents/3rd AY/2nd sem/thesis/dataset.txt")
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
    clustered_ngrams_str = {}
    for n in sorted(clustered_ngrams):
        clustered_ngrams_str[n] = '\n'.join(clustered_ngrams[n])
        print(f"\nClustered {n}-grams:\n")
        for ngram in clustered_ngrams[n]:
            print(ngram)

    # Compute scores
    mlm_scores_bert, mposm_scores_bert = zip(*[compute_mlm_and_mposm_scores(sentence, bert_model, bert_tokenizer) for sentence in lemmatized_sentences])
    mlm_scores_roberta, mposm_scores_roberta = zip(*[compute_mlm_and_mposm_scores(sentence, roberta_model, roberta_tokenizer) for sentence in lemmatized_sentences])

    # Extract patterns
    patterns = extract_patterns(lemmatized_sentences, mlm_scores_bert, mposm_scores_bert, mlm_scores_roberta, mposm_scores_roberta)
    print("\nEXTRACTED PATTERNS!\n")
    for pattern in patterns:
        print(pattern)

    # Save to Excel
    results = {
        "Sentences": sentences,
        "General_POS_Tagged": general_pos_tagged_sentences,
        "Detailed_POS_Tagged": detailed_pos_tagged_sentences,
        "Lemmatized_Sentences": lemmatized_sentences,
        "MLM_Scores_BERT": mlm_scores_bert,
        "MPOSM_Scores_BERT": mposm_scores_bert,
        "MLM_Scores_RoBERTa": mlm_scores_roberta,
        "MPOSM_Scores_RoBERTa": mposm_scores_roberta,
        "Patterns": patterns,
        "Clustered_NGrams": [clustered_ngrams_str]
    }

    df = pd.DataFrame.from_dict(results, orient='index').transpose()
    df.to_excel("results.xlsx", index=False)
    print("Results saved to results.xlsx")

if __name__ == "__main__":
    main()
