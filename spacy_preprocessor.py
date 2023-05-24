import re
from typing import List


import spacy
from spacy.tokens import Doc
from tqdm import tqdm

class SpaCyPreProcessor:

    def __init__(self, spacy_model = None, remove_numbers = False, remove_special = False, pos_to_remove = None, remove_stop_words = False, lemmatize = False, use_gpu = False) -> None:
        
        self.__remove_numbers = remove_numbers
        self.__remove_special = remove_special
        self.__pos_to_remove = pos_to_remove
        self.__remove_stop_words = remove_stop_words
        self.__lemmatize = lemmatize

        if spacy_model is None:
            self.model = spacy.load("en_core_web_sm")
        else:
            self.model = spacy_model

        if use_gpu:
            spacy.prefer_gpu()

    @staticmethod
    def download_spacy_model(model="en_core_web_sm"):
        print(f"Downloading spaCy model {model}")
        spacy.cli.download(model)
        print(f"Finished downloading model")

    @staticmethod
    def load_model(model="en_core_web_sm"):
        return spacy.load(model, disable=["ner", "parser"])

    def tokenize(self, text) -> List[str]:
        """
        Tokenize text using a spaCy pipeline
        :param text: Text to tokenize
        :return: list of str
        """
        doc = self.model(text)
        return [token.text for token in doc]

    def preprocess_text(self, text) -> str:
        """
        Runs a spaCy pipeline and removes unwanted parts from text
        :param text: text string to clean
        :return: str, clean text
        """
        doc = self.model(text)
        return self.__clean(doc)

    def preprocess_text_list(self, texts=List[str]) -> List[str]:
        """
        Runs a spaCy pipeline and removes unwantes parts from a list of text.
        Leverages spaCy's `pipe` for faster batch processing.
        :param texts: List of texts to clean
        :return: List of clean texts
        """
        clean_texts = []
        for doc in tqdm(self.model.pipe(texts)):
            clean_texts.append(self.__clean(doc))

        return clean_texts

    def __clean(self, doc: Doc) -> str:

        tokens = []
        # POS Tags removal
        if self.__pos_to_remove:
            for token in doc:
                if token.pos_ not in self.__pos_to_remove:
                    tokens.append(token)
        else:
            tokens = doc

        # Remove Numbers
        if self.__remove_numbers:
            tokens = [
                token for token in tokens if not (token.like_num or token.is_currency)
            ]

        # Remove Stopwords
        if self.__remove_stop_words:
            tokens = [token for token in tokens if not token.is_stop]
        # remove unwanted tokens
        tokens = [
            token
            for token in tokens
            if not (
                token.is_punct or token.is_space or token.is_quote or token.is_bracket
            )
        ]

        # Remove empty tokens
        tokens = [token for token in tokens if token.text.strip() != ""]

        # Lemmatize
        if self.__lemmatize:
            text = " ".join([token.lemma_ for token in tokens])
        else:
            text = " ".join([token.text for token in tokens])

        if self.__remove_special:
            # Remove non alphabetic characters
            text = re.sub(r"[^a-zA-Z\']", " ", text)
        # remove non-Unicode characters
        text = re.sub(r"[^\x00-\x7F]+", "", text)

        text = text.lower()

        return text

    
