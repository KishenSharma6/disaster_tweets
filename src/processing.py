import re
import pandas as pd

from nltk.corpus import stopwords


def tokenizer(input:pd):
    """
    Function takes pandas series and converts each row into tokens after removing punctuation,
    digits, stopwords, and capitalization

    Returns pd dataframe
    """
    cleaned_input = re.sub(r'[^\w\s]', '', str(input).lower()) #Remove punct
    cleaned_input = re.sub(r'[0-9]', '', cleaned_input) #remove numbers
    
    tokens = [word for word in cleaned_input.split(" ")]
    return tokens

def get_vocabulary(input:pd):
    """
    Takes tokens as inputs and returns an indexed dictionary of sorted vocabulary w/
    stopwords and empty strings removed
    """
    #create set of tokens
    vocab_set = set()

    for row in input:
        clean_tokens = [token for token in row if token not in stopwords.words('english')]
        for token in clean_tokens:
            if len(token) ==0:
                continue
            vocab_set.add(token)

    #create sorted dictionary w/ index
    vocabulary = {}

    for i in range(len(vocab_set)):
        temp = list(vocab_set)

        vocabulary[i] = temp[i]

    return vocabulary