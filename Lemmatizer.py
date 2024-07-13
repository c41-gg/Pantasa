import subprocess


def lemmatize(sentence):

    # Start the JVM with the JAR file in the classpath
    cmd = ['java', '-cp', '.:Modules/Morphinas/Morphinas.jar', 'Stemmer.Stemmer']

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
   
    stdout, stderr = process.communicate(sentence)

    if stderr:
        print(f"Error: {stderr}")
        return None
    
    return stdout.strip()

if __name__ == "__main__":
    sentence = "Ito ay isang halimbawa ng pangungusap."
    result = lemmatize(sentence)
    print(result)

