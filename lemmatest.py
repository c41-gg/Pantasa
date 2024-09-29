import calamancy
nlp = calamancy.load("tl_calamancy_md-0.1.0")
doc = nlp("Ako si Juan de la Cruz")



def lemmatize_sentence(sentence):
    """
    Calls the nlu lemmatizer to lemmatize a sentence and returns the lemmatized string.
    """
    try:
        # Check if the sentence is enclosed in single quotation marks with a comma before the closing mark
        if sentence.startswith('"') and sentence.endswith('"') and sentence.__contains__(','):
            sentence = sentence[0:-1]  # Remove the opening and closing quotation marks and the comma

        # Load the lemmatizer model
        parser = Parser("tl_calamancy_md-0.1.0")

        # Use the model to predict lemmatized output
        result = nlp.load(sentence)


        # Parse a sentence
        dependencies = list(parser(sentence))

        # Output the tokens and their dependency relations
        for token, dep in dependencies:
            print(f"Token: {token}, Dependency: {dep}")


        # Add back the quotation marks if they were removed
        if ',' in sentence:
            lemmatized_string = '"' + lemmatized_string + '"'

        return lemmatized_string
        
    except Exception as e:
        print(f"Exception occurred during lemmatization: {e}")
        return sentence


