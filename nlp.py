#Text Processing libraries
import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

def Tokenize(string):
    # Normalize
    normalized = re.sub(r"[^a-zA-Z0-9]", " ", string.lower().strip())
    # Tokenize the string into a list
    words = word_tokenize(normalized)
    # Remove stop words: if a token is a stop word, then remove it
    words = [w for w in words if w not in stopwords.words("english")]
    # Lemmatize and Stemming
    lemmed_words = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return " ".join(lemmed_words)


def nlp(contents):
    for i, text in enumerate(contents):
        contents[i] = Tokenize(text)
    return contents
