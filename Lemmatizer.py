import subprocess
import csv
import re
import string
import os

def lemmatize_sentence(sentence):
    """
    Calls the stemmer.py script to lemmatize a sentence and returns the lemmatized string.
    Ignores punctuation and numbers, leaving them unchanged.
    """
    mode = "2"  # Assuming raw string mode
    info_dis = "0"  # Assuming no info display

    # Remove punctuation and numbers, split sentence into tokens
    translator = str.maketrans('', '', string.punctuation + string.digits)
    cleaned_sentence = sentence.translate(translator)

    try:
        # Call stemmer.py script with cleaned sentence as source
        result = subprocess.run(
            ["python", "Modules/TagalogStemmerPython-master/TglStemmer.py", mode, cleaned_sentence, info_dis],
            capture_output=True, text=True, encoding='utf-8', check=True
        )

        # Check if there's any output
        if result.stdout:
            # Use regular expressions to extract root values
            root_values = re.findall(r"'root': '([^']+)'", result.stdout)
            lemmatized_string = ' '.join(root_values)  # Combine all roots into a single string
            return lemmatized_string  # Return only the lemmatized string
        else:
            print("Error: No output from stemmer script.")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Error lemmatizing sentence: {e}")
        print(e.stderr)  # Print stderr for debugging
        return None
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None

def process_csv_with_lemmatizer(input_csv, input_column, output_column):
    """
    Process a CSV file by lemmatizing the data in the input_column and appending results to output_column.
    """
    output_csv = 'output.csv'  # Temporary output file

    with open(input_csv, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        # Check if output_column already exists, otherwise add it
        if output_column not in fieldnames:
            fieldnames.append(output_column)

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if input_column in row:
                input_data = row[input_column]
                # Example processing (lemmatization)
                processed_data = lemmatize_sentence(input_data)
                row[output_column] = processed_data

            writer.writerow(row)

    # Replace original file with processed file
    os.replace(output_csv, input_csv)

# Example usage
if __name__ == "__main__":
    input_csv = 'database/preprocessed.csv'  # Replace with your input CSV file path
    input_column = 'Sentences'  # Replace with the column containing sentences to lemmatize
    output_column = 'Lemmatized'  # Replace with the column name where lemmatized results will be stored

    # Check if input CSV file exists
    if not os.path.exists(input_csv):
        print(f"Error: The file {input_csv} does not exist.")
    else:
        process_csv_with_lemmatizer(input_csv, input_column, output_column)
