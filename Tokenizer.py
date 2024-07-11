import re

def tokenize(text):
    # Regular expression to match sentences
    regex = r"[^.!?,\s][^.!?,]*(?:[.!?,](?!['\"]?\s|$)[^.!?,]*)*[.!?,]?['\"]?(?=\s|$)"
    pattern = re.compile(regex)
    
    sentences = pattern.findall(text)
    return sentences
