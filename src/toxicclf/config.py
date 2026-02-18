import torch
from dataclasses import dataclass

@dataclass
class Config:
    
    # Data parameters
    max_vocab_size: int = 20000
    min_freq: int = 2
    max_seq_len: int = 200
    
    # Model parameters
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    
    # Training parameters
    batch_size: int = 128
    epochs: int = 1
    learning_rate: float = 1e-3
    
    # Other parameters
    test_size: float = 0.2
    seed: int = 42
    vocab_path: str = "artifacts/vocab.json"
    save_checkpoint_dir: str = "artifacts/checkpoints"
    saved_model_name: str = "best_model.pt"
    artifacts_dir: str = "artifacts"
    
    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"