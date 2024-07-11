from Tokenizer import tokenize
from POSDTagger import pos_tag as pos_gtag
from POSGTagger import pos_tag  as pos_dtag

def main():
    text = input("Input text here: ")
    
    sentences = tokenize(text)
    print("\nTOKENIZED VERSION!\n") 
    for sentence in sentences:
        print(sentence)
    
    print("\nGENERAL POS TAGGED VERSION!\n")
    for sentence in sentences:
        tagged_sentence = pos_gtag(sentence)
        print(tagged_sentence)
        print("\n")

    print("\nDETAILED POS TAGGED VERSION!\n")
    for sentence in sentences:
        tagged_sentence = pos_dtag(sentence)
        print(tagged_sentence)
        print("\n")
    

if __name__ == "__main__":
    main()