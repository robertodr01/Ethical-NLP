import pandas as pd
import re
from curses.ascii import isdigit
from utils.spacy_preprocessor import SpaCyPreProcessor

def extract_emfd(filename) -> tuple[dict, dict]:
    
    emfd = dict()
    emfd_categories = dict()

    with open(filename, "r") as file:
    
        for line in file.readlines():
            if line[0] != "%" and line[0] != "\n" and not line[0].isdigit(): 
                key, value = line.split("\t")
                emfd[key] = int(value)
            if line[0].isdigit():
                key, value = line.split("\t")
                emfd_categories[key] = value.removesuffix("\n")

    return emfd_categories, emfd

def extract_lemmatized_emfd(spacy_pipeline, emfd_dict) -> dict:
    spacy_model = SpaCyPreProcessor.load_model(spacy_pipeline)
    preprocessing_pipeline = SpaCyPreProcessor(spacy_model=spacy_model, remove_numbers=True, remove_special=True, remove_stop_words=True, lemmatize=True, use_gpu=True)

    emfd_lemma = dict()

    for key, value in emfd_dict.items():
        key_lemma = preprocessing_pipeline.preprocess_text(key)
        if key_lemma in emfd_lemma:
            emfd_lemma[key_lemma] = emfd_lemma[key_lemma].union({value})
        else:
            emfd_lemma[key_lemma] = {value}

    return emfd_lemma
