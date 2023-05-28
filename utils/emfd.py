import pandas as pd
import re
from curses.ascii import isdigit

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
