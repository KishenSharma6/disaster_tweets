import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import re
import pandas as pd

from nltk.corpus import stopwords

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0:"<PAD>", 1: "<SOS>", 2:"<EOS>", 3:"<UNK>"} #index 2 string
        #UNK token for words that do not meet threshold requirement
        self.stoi = {"<PAD>":0, "<SOS>": 1, "<EOS>":2, "<UNK>":3} #string to index// inverse of itos
        self.freq_threshold= freq_threshold
    
    def __len__(self):
        return len(self.itos) #get length of vocabulary
    
    @staticmethod
    def tokenizer(text):
        cleaned_text = re.sub(r'[^\w\s]', '', str(text).lower()) #Remove punct
        cleaned_text = re.sub(r'[0-9]', '', cleaned_text) #remove numbers

        tokens = [word for word in cleaned_text.split(" ")]
        return tokens

    def build_vocabulary(self, sentence_list):
        frequencies= {}
        idx = 4 #idx 0-3 taken by our tokens
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                if word in frequencies:
                    frequencies[word] +=1
                else:
                    frequencies[word] = 1
                #if word meets our frequency requirement, we will add to vocab    
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word #stoi and itos are inverses of one another
                    idx +=1

    def numericalize(self, text):
        #convert text to numerical value
        tokens = self.tokenizer(text)
        #code words that dont meet freq requirement with unk value
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokens
        ]

class Dataset(Dataset):
    def __init__(self, data, targets, freq_threshold = 2) -> None:
        self.data = data
        self.targets = targets
        
    # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold) #check out how the inheritance is 
                                                # done in this class
        self.vocab.build_vocabulary(self.data.tolist())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.targets[idx]

        #Conver text to numbers(based on index in vocabulary)
        numericalized_text = [self.vocab.stoi["<SOS>"]] #stringTOindex // get start of sentence
        numericalized_text += self.vocab.numericalize(text) #change word to index in vocab
        numericalized_text.append(self.vocab.stoi["<EOS>"])

        return torch.tensor(numericalized_text), torch.tensor(label)

class MyCollate:
    #we'll use this with data loader to pad our data
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        text = [item[0].unsqueeze(0) for item in batch]
        padded_text = pad_sequence(text, batch_first=False, padding_value=self.pad_idx)

        labels = [item[1] for item in batch]
        labels = torch.cat(labels, dim=0)

        return padded_text, labels