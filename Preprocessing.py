from Tokenizer import tokenize
from POSDTagger import pos_tag as pos_dtag
from POSRTagger import pos_tag  as pos_rtag

def main():
    text = input("Input text here: ")
    
    sentences = tokenize(text)
    print("\nTOKENIZED VERSION!\n") 
    for sentence in sentences:
        print(sentence)
    
    print("\nGENERAL POS TAGGED VERSION!\n")
    for sentence in sentences:
        tagged_sentence = pos_rtag(sentence)
        print(tagged_sentence)

    print("\nDETAILED POS TAGGED VERSION!\n")
    for sentence in sentences:
        tagged_sentence = pos_dtag(sentence)
        print(tagged_sentence)
    

if __name__ == "__main__":
    main()