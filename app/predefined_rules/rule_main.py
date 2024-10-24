import pandas as pd
import sys
import os
from app.predefined_rules.hyphen_rule import correct_hyphenation
from app.predefined_rules.rd_rule import rd_interchange
import logging
from app.preprocess import pos_tagging

logger = logging.getLogger(__name__)

# Add the 'app' directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the necessary 
from rules.Modules.POSDTagger import pos_tag

# Hierarchical POS Tag Dictionary
hierarchical_pos_tags = {
    "NN.*": ["NNC", "NNP", "NNPA", "NNCA"],
    "PR.*": ["PRS", "PRP", "PRSP", "PRO", "PRQ", "PRQP", "PRL", "PRC", "PRF", "PRI"],
    "DT.*": ["DTC", "DTCP", "DTP", "DTPP"],
    "CC.*": ["CCT", "CCR", "CCB", "CCA", "CCP", "CCU"],
    "LM": [],
    "TS": [],
    "VB.*": ["VBW", "VBS", "VBH", "VBN", "VBTS", "VBTR", "VBTF", "VBTP", "VBAF", "VBOF", "VBOB", "VBOL", "VBOI", "VBRF"],
    "JJ.*": ["JJD", "JJC", "JJCC", "JJCS", "JJCN", "JJN"],
    "RB.*": ["RBD", "RBN", "RBK", "RBP", "RBB", "RBR", "RBQ", "RBT", "RBF", "RBW", "RBM", "RBL", "RBI", "RBJ", "RBS"],
    "CD.*": ["CDB"],
    "FW": [],
    "PM.*": ["PMP", "PME", "PMQ", "PMC", "PMSC", "PMS"]
}

def load_dictionary(csv_path):
    df = pd.read_csv(csv_path)
    # Assuming the words are in the first column and filtering out non-string values
    words = df.iloc[:, 0].dropna().astype(str).tolist()
    return words

def handle_nang_ng(text, pos_tags):
    vowels = 'aeiou'  # Define vowels
    words = text.split()
    corrected_words = []
    
    for i, word in enumerate(words):
        if word == "ng":
            # Handle "ng" as a ligature connecting verbs to adverbs
            if i > 0 and pos_tags[i - 1].startswith('VB') and pos_tags[i + 1].startswith('RB'):
                corrected_words.append("ng")
                print("'ng' kept as ligature (connecting verb and adverb)")
            
            # Handle "ng" as other use cases, such as between nouns and adjectives
            elif i > 0 and pos_tags[i + 1].startswith('JJ'):
                corrected_words.append("ng")
                print("'ng' kept as ligature (connecting noun and adjective)")
            
            # Other cases where "ng" can be corrected to "nang" based on POS
            else:
                corrected_words.append(word)

        elif word == "nang":
            # Case 1: "nang" as a coordinating conjunction
            if pos_tags[i] == 'CCB' and i > 0:
                corrected_words.append("ng")
                print("'nang' corrected to 'ng' (coordinating conjunction CCB)")
            
            # Case 2: Ligature use (connecting an adjective or a verb to a noun/adjective/adverb)
            elif i > 0 and pos_tags[i - 1] in ['JJ', 'VB']:  # Check if the previous word is an adjective (JJ) or a verb (VB)
                corrected_words.append("ng")
                print("'nang' corrected to 'ng' (ligature for adjective/verb)")
            
            # Case 3: Check if "nang" is followed by an adjective (often requires ligature)
            elif i < len(pos_tags) - 1 and pos_tags[i + 1].startswith('JJ'):
                corrected_words.append("ng")
                print("'nang' corrected to 'ng' (followed by adjective JJ)")
            
            # Case 4: Check if "nang" is tagged as adverb (RBW) but acts as a ligature
            elif pos_tags[i] == 'RBW' and i > 0 and pos_tags[i - 1] in ['NNC', 'NNP']:
                corrected_words.append("ng")
                print("'nang' corrected to 'ng' (acting as ligature with noun)")
            
            # Case 5: Other valid use cases (e.g., adverbial phrases)
            else:
                corrected_words.append(word)
                print("'nang' kept unchanged (other use cases)")
        
        elif word == "na" and i > 0:  # If the current word is "na" and it's not the first word
            prev_word = words[i - 1]
            print(f"Checking if 'na' should be merged with the previous word: {prev_word}")
            
            if prev_word[-1].lower() in vowels:  # Check if the previous word ends with a vowel
                corrected_word = prev_word + "ng"
                corrected_words[-1] = corrected_word  # Update the last word in corrected_words
                print(f"Word ending with vowel: Merged 'na' to form '{corrected_word}'")

            elif prev_word[-1].lower() == 'n':  # Check if the previous word ends with 'n'
                corrected_word = prev_word + "g"
                corrected_words[-1] = corrected_word  # Update the last word in corrected_words
                print(f"Word ending with 'n': Merged 'na' to form '{corrected_word}'")
            
        else:
            corrected_words.append(word)  # Append the word if no correction is made
            print(f"No correction needed for '{word}'")
    
    corrected_text = ' '.join(corrected_words)
    print(f"\nFinal corrected text: {corrected_text}")
    return corrected_text

# Load the dictionary from the CSV file
dictionary_file = load_dictionary("data/raw/dictionary.csv")

def separate_mas(text, dictionary=dictionary_file):
    words = text.split()
    for i, word in enumerate(words):
        if word.lower().startswith("mas"):
            if word not in dictionary:
                remaining = word[3:]
                words[i] = "mas " + remaining
    return ' '.join(words)

affix_list = {
    "common_prefixes": ["pinaka","pinang", "nag", "na", "mag", "ma", "i", "i-", "ika-", "isa-", "ipag", "ipang", "ipa", "pag", "pa", "um", "in", "ka", "ni", "pinaka", "pinag", "pina"],
    "prefix_assimilation": ["pang", "pam", "pan", "mang", "mam", "man", "sang", "sam", "san", "sing", "sim", "sin"],
    "common_infixes": ["um", "in"],
    "common_suffixes": ["nan", "han", "hin", "an", "in", "ng"],
    "long_prefixes": ["napag", "mapag", "nakipag", "nakikipag", "makipag", "makikipag", "nakiki", "makiki", "naka", "nakaka"],
    "compound_prefix": ["pinag", "pinagpa", "ipang", "ipinag", "ipinagpa", "nakiki", "makiki", "nakikipag", "napag", "mapag", "nakipag", "makipag", "naka", "maka", "nagpa", "nakaka", "makaka", "nagka", "nagkaka", "napaki", "napakiki", "mapaki", "mapakiki", "paki", "pagka", "pakiki", "pakikipag", "pagki", "pagkiki", "pagkikipag", "ika", "ikapag", "ikapagna", "ikima", "ikapang", "ipa", "ipaki", "ipag", "ipagka", "ipagpa", "ipapang", "makapag", "magkanda", "magkang", "makasing", "maging", "maging", "nakapag", "napaka"]
}

# Combine all prefixes for easy access
prefixes = affix_list["common_prefixes"] + affix_list["long_prefixes"] + affix_list["prefix_assimilation"] + affix_list["compound_prefix"]

merge_affix_pos = hierarchical_pos_tags['VB.*'] + hierarchical_pos_tags['NN.*'] + hierarchical_pos_tags['JJ.*']

def merge_affixes(text, pos_tags, dictionary=dictionary_file):
    words = text.split()
    corrected_words = []
    i = 0

    while i < len(words):
        word = words[i]
        merged = False
        
        # Check if the word is an affix
        if word.lower() in prefixes:
            # Ensure there is a next word to merge with
            if i + 1 < len(words):
                next_word = words[i + 1]
                next_word_pos = pos_tags[i + 1]  # Get the POS tag of the next word
                
                # Only merge if the next word's POS tag is in mergee_affix_pos list
                if next_word_pos in merge_affix_pos:
                    # Merge affix with the next word
                    combined_word = word + next_word
                    # Check if the combined word exists in the dictionary
                    if combined_word.lower() in dictionary:
                        corrected_words.append(combined_word)
                        i += 2  # Skip the next word as it's already merged
                        merged = True
        
        if not merged:
            corrected_words.append(word)
            i += 1
    
    return ' '.join(corrected_words)

def apply_predefined_rules_pre(text):
    pos = pos_tag(text)
    rd_correction = rd_interchange(text)
    mas_correction = separate_mas(rd_correction)
    rule_corrected = handle_nang_ng(mas_correction,pos)

    return rule_corrected

def apply_predefined_rules_post(text):
    pos = pos_tag(text)

    
    prefix_merged = merge_affixes(text, pos)
    rule_corrected = correct_hyphenation(prefix_merged)

    return rule_corrected

def apply_predefined_rules(text):
    pos = pos_tag(text)

    mas_correction = separate_mas(text, pos)
    prefix_merged = merge_affixes(mas_correction, pos)
    nang_handled = handle_nang_ng(prefix_merged, pos)
    rule_corrected = correct_hyphenation(nang_handled)

    logger.info(f"Final rule-corrected sentence: {rule_corrected}")

    return rule_corrected


if __name__ == "__main__":
    text = "pinang tiklop maski pagusp"
    corrected_sentence = apply_predefined_rules_pre(text)

    print(f"Corrected sentence: {corrected_sentence}")

    # Test case
    test = "tumakbo ng mabilis"
    pos = pos_tag(test).split()  # Split the string into a list
    print(test)
    print(pos)
    print(handle_nang_ng(test, pos))
