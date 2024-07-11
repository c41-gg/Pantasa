import re

def tokenize(text):
    # Regular expression to match sentences
    regex = r"[^.!?,\s][^.!?,]*(?:[.!?,](?!['\"]?\s|$)[^.!?,]*)*[.!?,]?['\"]?(?=\s|$)"
    pattern = re.compile(regex)
    
    sentences = pattern.findall(text)
    return sentences

def main():
    text = input("input text here: ")
    
    sentences = tokenize(text)
    print ("\nTOKENIZED VERSION!\n") 
    for sentence in sentences:
        print(sentence)

if __name__ == "__main__":
    main()
