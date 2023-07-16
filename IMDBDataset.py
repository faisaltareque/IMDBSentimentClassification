import spacy
import torch
from collections import Counter
from torchtext.vocab import vocab
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# Dwnload the dataset
# !wget https://github.com/rasbt/python-machine-learning-book-3rd-edition/raw/master/ch08/movie_data.csv.gz
# !gunzip -f movie_data.csv.gz 

class IMDBDataset(Dataset):
    def __init__(self, dataframe, tokenizer, min_freq, vocabulary=None):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.min_freq = min_freq
        if vocabulary is not None:
            self.vocab = vocabulary
        else:
            self.counter = Counter()
            for item in self.df['review']:
                self.counter.update(self.tokenizer(item))
            self.vocab = vocab(self.counter, min_freq=self.min_freq, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))
            self.vocab.set_default_index(self.vocab['<unk>'])
        self.text_transform = lambda x: [self.vocab['<BOS>']] + [self.vocab[token] for token in tokenizer(x)] + [self.vocab['<EOS>']]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.text_transform(self.df.iloc[idx]['review'])
        label = self.df.iloc[idx]['sentiment']
        return torch.tensor(text), torch.tensor(label)
    
class IMDBCollate:
    def __init__(self, pad_idx, batch_first=True):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
        
    def __call__(self, batch):
        label_list, text_list = [], []
        for (_text, _label) in batch:
            label_list.append(_label)
            text_list.append(_text)
        return pad_sequence(text_list, padding_value=self.pad_idx, batch_first=self.batch_first), torch.tensor(label_list)
