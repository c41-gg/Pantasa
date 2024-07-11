import subprocess

# Mapping of detailed POS tags to general POS tags
tag_mapping = {
    "NNC": "NN.*", "NNP": "NN.*", "NNPA": "NN.*", "NNCA": "NN.*",
    "PRS": "PR.*", "PRP": "PR.*", "PRSP": "PR.*", "PRO": "PR.*",
    "PRQ": "PR.*", "PRQP": "PR.*", "PRL": "PR.*", "PRC": "PR.*",
    "PRF": "PR.*", "PRI": "PR.*", "DTC": "DT.*", "DTCP": "DT.*",
    "DTP": "DT.*", "DTPP": "DT.*", "CCT": "CC.*", "CCR": "CC.*",
    "CCB": "CC.*", "CCA": "CC.*", "CCP": "CC.*", "CCU": "CC.*",
    "VBW": "VB.*", "VBS": "VB.*", "VBH": "VB.*", "VBN": "VB.*",
    "VBTS": "VB.*", "VBTR": "VB.*", "VBTF": "VB.*", "VBTP": "VB.*",
    "VBAF": "VB.*", "VBOF": "VB.*", "VBOB": "VB.*", "VBOL": "VB.*",
    "VBOI": "VB.*", "VBRF": "VB.*", "JJD": "JJ.*", "JJC": "JJ.*",
    "JJCC": "JJ.*", "JJCS": "JJ.*", "JJCN": "JJ.*", "JJN": "JJ.*",
    "RBD": "RB.*", "RBN": "RB.*", "RBK": "RB.*", "RBP": "RB.*",
    "RBB": "RB.*", "RBR": "RB.*", "RBQ": "RB.*", "RBT": "RB.*",
    "RBF": "RB.*", "RBW": "RB.*", "RBM": "RB.*", "RBL": "RB.*",
    "RBI": "RB.*", "RBJ": "RB.*", "RBS": "RB.*", "CDB": "CD.*",
    "PMP": "PM.*", "PME": "PM.*", "PMQ": "PM.*", "PMC": "PM.*",
    "PMSC": "PM.*", "PMS": "PM.*", "LM": "LM", "TS": "TS",
     "FW":"FW"
}

def map_tag(tag):
    # Check for combined tags
    if "_" in tag:
        parts = tag.split("_")
        for part in parts:
            if part in tag_mapping:
                return tag_mapping[part]
    return tag_mapping.get(tag, "X")

def pos_tag(sentence):
    # Set the path to the Stanford POS Tagger directory
    stanford_pos_tagger_dir = "C:/Users/Carlo Agas/Documents/GitHub/Pantasa/Modules/FSPOST"

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

     # Process the output and map detailed tags to general tags
    tagged_sentence = result.stdout.strip()
    general_tagged_sentence = []

    for word_tag in tagged_sentence.split('\n'):
        try:
            word, tag = word_tag.split('\t')
            general_tag = map_tag(tag)  # Use the map_tag function to handle combined tags
            general_tagged_sentence.append(f"{word}/{general_tag}")
        except ValueError:
            # Skip lines that do not have exactly two parts
            continue

    return " ".join(general_tagged_sentence)

# Example usage
sentence = "Ang mga tao ay may iba't ibang wika at kultura. Ang BERT model ay ginagamit para sa natural language processing. Si Juan ay mahilig magbasa ng mga libro sa kanyang libreng oras. Ang Pilipinas ay isang arkipelago na matatagpuan sa Timog-Silangang Asya. Ang RoBERTa ay isang variant ng BERT na mas mahusay sa ilang mga task."
tagged_sentence = pos_tag(sentence)
print("Tagged Sentence:", tagged_sentence)
