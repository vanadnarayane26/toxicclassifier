import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from toxicclf.utils.logger import get_logger
from toxicclf.utils.setup_nltk import setup_nltk
logger = get_logger(__name__)
setup_nltk()

class Preprocessor:
    def __init__(self, stop_words:bool = True, lemmatize:bool = False):
        self.stop_words = stop_words
        self.lemmatize = lemmatize
        if self.stop_words:
            self.stop_words_set = set(stopwords.words('english'))
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text:str):
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers
        text = re.sub(r'[^A-Za-z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text:str):
        tokens = word_tokenize(text)
        
        # Remove stop words if enabled
        if self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words_set]
        
        self.text = ' '.join(tokens)
        # Lemmatize if enabled
        if self.lemmatize:
            tokens = self.text.split()
            self.text = ' '.join([self.lemmatizer.lemmatize(token) for token in tokens])
        return tokens
    
    def preprocess(self, text:str):
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        return tokens
        