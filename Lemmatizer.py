import subprocess

def lemmatize_sentence(sentence):
    try:
        result = subprocess.run(
            ['java', '-cp', 'Modules/Morphinas/Morphinas.jar', 'Stemmer.Stemmer', 'lemmatizeSentence', sentence],
            capture_output=True, text=True, check=True
        )
        
        # Print Java program output for debugging
        print("Java Program Output:", result.stdout)
        
        return result.stdout.strip()
    
    except subprocess.CalledProcessError as e:
        print("Error executing Java program:", e)
        if e.stderr:
            print("Java Program Error Output:", e.stderr)
        return None

# Example usage
sentence = "Paggamot ng mga worm sa binti katutubong remedyong Panandalian iwas sa helmint"
lemmatized_sentence = lemmatize_sentence(sentence)

if lemmatized_sentence:
    print(f"Lemmatized Sentence: {lemmatized_sentence}")
else:
    print("Failed to lemmatize sentence.")
