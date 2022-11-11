from lib2to3.pgen2.token import tok_name
import torch


class Dataset:
    def __init__(self, data, targets, tokenizer) -> None:
        self.data = data
        self.targets = targets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.targets[idx]

        #input_ids are tokens
        input_ids = self.tokenizer(text)

        #Padding
        
        return {
            "text": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label)
        }