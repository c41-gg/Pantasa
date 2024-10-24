#main function is correct_hyphenation(text)

import re
import pandas as pd

def load_dictionary(csv_path):
    df = pd.read_csv(csv_path)
    # Assuming the words are in the first column and filtering out non-string values
    words = df.iloc[:, 0].dropna().astype(str).tolist()
    return words

# Load the dictionary from the CSV file
dictionary_file = load_dictionary("data/raw/dictionary.csv")

affix_list = {
    "common_prefixes": ["pinang", "nag", "na", "mag", "ma", "i", "i-", "ika-", "isa-", "ipag", "ipang", "ipa", "pag", "pa", "um", "in", "ka", "ni", "pinaka", "pinag", "pina"],
    "prefix_assimilation": ["pang", "pam", "pan", "mang", "mam", "man", "sang", "sam", "san", "sing", "sim", "sin"],
    "common_infixes": ["um", "in"],
    "common_suffixes": ["nan", "han", "hin", "an", "in", "ng"],
    "long_prefixes": ["napag", "mapag", "nakipag", "nakikipag", "makipag", "makikipag", "nakiki", "makiki", "naka", "nakaka"],
    "compound_prefix": ["pinag", "pinagpa", "ipang", "ipinag", "ipinagpa", "nakiki", "makiki", "nakikipag", "napag", "mapag", "nakipag", "makipag", "naka", "maka", "nagpa", "nakaka", "makaka", "nagka", "nagkaka", "napaki", "napakiki", "mapaki", "mapakiki", "paki", "pagka", "pakiki", "pakikipag", "pagki", "pagkiki", "pagkikipag", "ika", "ikapag", "ikapagna", "ikima", "ikapang", "ipa", "ipaki", "ipag", "ipagka", "ipagpa", "ipapang", "makapag", "magkanda", "magkang", "makasing", "maging", "maging", "nakapag", "napaka"]
}

# Combine all prefixes for easy access
prefixes = affix_list["common_prefixes"] + affix_list["long_prefixes"] + affix_list["prefix_assimilation"] + affix_list["compound_prefix"]


def is_consonant(char):
    """ Check if the character is a consonant. """
    return char.lower() in "bcdfghjklmnpqrstvwxyz"

def is_vowel(char):
    """ Check if the character is a vowel. """
    return char.lower() in "aeiou"

def detect_onomatopoeia(word):
    """
    Detect onomatopoeia, handling both hyphenated and non-hyphenated consonant-vowel pairs.
    Example: "tik-tak", "ding-dong", "plip-plap", "rat-ta-ta", and "taktak" -> "tak-tak".
    """
    # Handle the hyphenated onomatopoeia case (e.g., "tik-tak", "ding-dong")
    if '-' in word:
        parts = word.split('-')
        if all(len(part) == 3 for part in parts) and len(parts) in [2, 3]:  # Handle patterns like "tik-tak" and "rat-ta-ta"
            if all(is_consonant(part[0]) and is_vowel(part[1]) and is_consonant(part[2]) for part in parts):
                return True
    # Handle attached consonant-vowel pairs (e.g., "taktak" -> "tak-tak")
    if len(word) == 6 and word[:3] == word[3:]:
        if is_consonant(word[0]) and is_vowel(word[1]) and is_consonant(word[2]):
            return True

    return False

def detect_onomatopoeia_two_words(word1, word2):
    """
    Detect onomatopoeia when consonant-vowel pairs are separated into two words (e.g., "tik tak", "ding dong").
    """
    if len(word1) == 3 and len(word2) == 3:
        if is_consonant(word1[0]) and is_vowel(word1[1]) and is_consonant(word1[2]) and \
           is_consonant(word2[0]) and is_vowel(word2[1]) and is_consonant(word2[2]):
            return True
    return False

def separate_prefix_from_word(word):
    """ Separate consonant from vowel if a prefix ending in a consonant is added to a word starting with a vowel. """
    for prefix in prefixes:
        if word.startswith(prefix):
            remaining_word = word[len(prefix):]
            if remaining_word and is_vowel(remaining_word[0]) and is_consonant(prefix[-1]):
                # Insert hyphen between consonant (prefix end) and vowel (word start)
                return f"{prefix}-{remaining_word}", 1
            elif remaining_word.istitle():
                # Insert hyphen between prefix and the proper noun (remain_word)
                return f"{prefix}-{remaining_word}", 2
            else:
                return f"{prefix}{remaining_word}", 0

def correct_hyphenation(text, dictionary=dictionary_file):
    words = text.split()  # Split the text into words
    corrected_words = []  # List to store the corrected words
    i = 0  # Initialize the word index

    while i < len(words):
        word = words[i]  # Get the current word
        print(f"\nEvaluating word: {word}")  # Log the current word being evaluated

        # Handle repeating words like "ano-ano"
        if i < len(words) - 1 and words[i] == words[i + 1]:
            print(f" - Repeated word without hyphen detected: {word}, adding hyphen.")
            corrected_words.append(f"{word}-{word}")
            i += 1  # Skip the next repeated word

        # Handle compound words like "ika8"
        elif re.match(r'ika\d+', word):
            print(f" - Compound detected: {word}, adding hyphen to form 'ika-8'.")
            corrected_words.append(re.sub(r'(ika)(\d+)', r'\1-\2', word))

        # Handle cases like "ika 8" (separated "ika" and number)
        elif word == "ika" and i < len(words) - 1 and re.match(r'\d+', words[i + 1]):
            print(f" - Compound detected: 'ika' and '{words[i + 1]}', joining them as 'ika-{words[i + 1]}'.")
            corrected_words.append(f"ika-{words[i + 1]}")
            i += 1  # Skip the number part since it's now joined with "ika-"

        # Handle hyphenation after "de" and "di" (like "dimahal" or "dekalidad")
        elif word.startswith("de") or word.startswith("di"):
            root_word = word[2:]  # Remove the "de" or "di" prefix
            if root_word.lower() in dictionary:
                # If the root word is valid, add the hyphen
                corrected_word = f"{word[:2]}-{word[2:]}"
                print(f" - Prefix '{word[:2]}' detected, adding hyphen: {corrected_word}")
                corrected_words.append(corrected_word)
            else:
                # If root word is not in dictionary, no changes
                print(f" - Word '{word}' not found in dictionary, no changes.")
                corrected_words.append(word)

        # Handle hyphenation of a prefix from the word
        elif any(word.startswith(prefix) for prefix in prefixes) and word not in dictionary:
            separated_word, case = separate_prefix_from_word(word)
            if separated_word != word:
                if case == 1:  # Prefix hyphenated from a proper noun
                    print(f" - Prefix with proper noun detected, corrected to '{separated_word}'.")
                if case == 2:  # Prefix that ends in a consonant hyphenated from word starting with vowel
                    print(f" - Separating consonant from vowel for prefix '{word}', corrected to '{separated_word}'.")
            corrected_words.append(separated_word)


        # Handle onomatopoeia (hyphenated and non-hyphenated)
        elif detect_onomatopoeia(word):
            if '-' not in word:  # Handle non-hyphenated patterns like "taktak"
                print(f" - Non-hyphenated onomatopoeia detected: {word}, converting to hyphenated form.")
                corrected_words.append(f"{word[:3]}-{word[3:]}")
            else:
                print(f" - Onomatopoeia detected: {word}, keeping it as is.")
                corrected_words.append(word)

        # Handle onomatopoeia with two separate words (e.g., "tik tak")
        elif i < len(words) - 1 and detect_onomatopoeia_two_words(words[i], words[i + 1]):
            print(f" - Onomatopoeia detected across two words: '{words[i]} {words[i + 1]}', adding hyphen.")
            corrected_words.append(f"{words[i]}-{words[i + 1]}")
            i += 1  # Skip the next word since it's now joined with the current one


        # Check for compound words with hyphen and validate if the combined word exists
        elif '-' in word:
            # Remove hyphen and check if combined word exists in the dictionary
            combined_word = word.replace('-', '')
            if combined_word in dictionary:
                print(f" - Combined version of '{word}' exists: {combined_word}, using the combined form.")
                corrected_words.append(combined_word)  # Use the combined word if it exists
            else:
                print(f" - Combined version of '{word}' does not exist, keeping the hyphenated form.")
                corrected_words.append(word)  # Keep the hyphenated form

        # Handle other cases where the word should remain unchanged
        else:
            print(f" - No special handling for: {word}, keeping as is.")
            corrected_words.append(word)

        # Always move to the next word unless we explicitly skip ahead
        i += 1

    return ' '.join(corrected_words)

