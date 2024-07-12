import subprocess

def pos_tag(sentence):
    # Set the path to the Stanford POS Tagger directory
    stanford_pos_tagger_dir = "Modules/FSPOST"

    # Set the paths to the model and jar files
    model = stanford_pos_tagger_dir + '/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    jar = stanford_pos_tagger_dir + '/stanford-postagger.jar'

    # Command to run the POS tagger
    command = [
        'java', '-mx300m', '-cp', jar, 'edu.stanford.nlp.tagger.maxent.MaxentTagger',
        '-model', model, '-outputFormat', 'tsv'
    ]

    # Run the command using subprocess
    result = subprocess.run(command, input=sentence, text=True, capture_output=True)

    if result.returncode != 0:
        print("Error:", result.stderr)
        return None

    # Process the output
    tagged_sentence = result.stdout.strip().split('\n')
    words = []
    pos_tags = []

    for word_tag in tagged_sentence:
        try:
            word, tag = word_tag.split('\t')
            words.append(word)
            pos_tags.append(tag)
        except ValueError:
            # Skip lines that do not have exactly two parts
            continue

    # Join POS tags into a single string similar to the input sentence
    pos_tag_sequence = " ".join(pos_tags)
    return pos_tag_sequence

