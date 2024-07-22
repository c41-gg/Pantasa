import re
from lingua import Language, LanguageDetectorBuilder

# Set up the language detector for Tagalog and English
languages = [Language.TAGALOG, Language.ENGLISH]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

def tokenize(text):
    # Regular expression to match sentences with strict punctuation delimiters, including em dash
    regex = r"[^.!?,;:—\s][^.!?,;:—]*[.!?,;:—]?['\"]?(?=\s|$)"
    pattern = re.compile(regex)
    
    # Find all sentences in the text
    sentences = pattern.findall(text)
    
    tagalog_sentences = []
    
    for sentence in sentences:
        detected_language = detector.detect_language_of(sentence)
        confidence = detector.compute_language_confidence(sentence, Language.TAGALOG)
        
        # Ensure the sentence is primarily Tagalog
        if detected_language == Language.TAGALOG and confidence > 0.98:
            # Check each word in the sentence to ensure no English word is detected
            words = sentence.split()
            is_taglish = False
            for word in words:
                word_language = detector.detect_language_of(word)
                if word_language == Language.ENGLISH:
                    is_taglish = True
                    break
            
            if not is_taglish:
                tagalog_sentences.append(sentence)
    
    return tagalog_sentences


