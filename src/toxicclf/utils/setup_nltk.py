import nltk
from toxicclf.utils.logger import get_logger
logger = get_logger(__name__)


def setup_nltk():
    # Map package names to the resource path used by nltk.data.find
    resources = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',  # Add this line
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4'
    }

    for package, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
            logger.info(f"NLTK package '{package}' is already available.")
        except LookupError:
            logger.info(f"Downloading NLTK package '{package}'...")
            nltk.download(package)
            logger.info(f"NLTK package '{package}' downloaded successfully.")
            
if __name__ == "__main__":
    setup_nltk()