import torch
from torch.utils.data import Dataset

from toxicclf.data.preprocessing import Preprocessor
from toxicclf.constants import LABEL_COLUMNS

class ToxicCommentsDataset(Dataset):
    def __init__(self, df, vocabulary, max_seq_len, preprocessor):
        self.df = df
        self.vocabulary = vocabulary
        self.max_seq_len = max_seq_len
        self.preprocessor = preprocessor
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = self.preprocessor.preprocess(row['comment_text'])
        encoded_tokens = self.vocabulary.word_to_index(tokens)
        
        encoded_tokens = encoded_tokens[:self.max_seq_len]
        padding_length = self.max_seq_len - len(encoded_tokens)
        if padding_length > 0:
            encoded_tokens.extend([0] * padding_length)
        labels = torch.tensor(row[LABEL_COLUMNS].values.astype(float), dtype=torch.float)
        
        return torch.tensor(encoded_tokens, dtype=torch.long), labels