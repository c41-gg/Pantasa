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
    tagged_sentence = result.stdout.strip()

    return tagged_sentence


