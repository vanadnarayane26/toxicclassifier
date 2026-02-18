import json
from pathlib import Path

def save_vocab(vocab, save_path):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(vocab.word2idx, f, ensure_ascii=False, indent=4)
        
def load_vocab(load_path):
    load_path = Path(load_path)
    if not load_path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {load_path}")
    with open(load_path, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word