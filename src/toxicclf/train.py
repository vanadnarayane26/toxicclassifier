from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from toxicclf.config import Config
from toxicclf.data.loader import Dataloader
from toxicclf.data.vocabulary import Vocabulary
from toxicclf.data.dataset import ToxicCommentsDataset
from toxicclf.data.preprocessing import Preprocessor    
from toxicclf.training.trainer import Trainer
from toxicclf.models.lstm import LSTMClassifier
from toxicclf.utils.logger import get_logger
from toxicclf.utils.io import save_vocab, load_vocab
logger = get_logger(__name__)


def run_training(train_path):
    path = Path(train_path)
    config = Config()
    logger.info("Starting training process...")
    logger.info(f"Loading data from: {path}")
    loader = Dataloader(path)
    df = loader.load_data()
    logger.info(f"Data loaded successfully. Number of samples: {len(df)}")
    
    train_df, val_df = train_test_split(df, test_size=config.test_size, random_state=config.seed)
    logger.info(f"Data split into train and validation sets. Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    logger.info("Starting preprocessing and vocabulary building...")
    
    preprocessor = Preprocessor()
    tokenized_texts = [preprocessor.preprocess(text) for text in train_df['comment_text']]
    vocabulary = Vocabulary(max_size=config.max_vocab_size, min_freq=config.min_freq)
    vocabulary.build_vocab(tokenized_texts)
    
    logger.info(f"Vocabulary built successfully. Vocabulary size: {len(vocabulary)}")
    logger.info("Saving vocabulary...")
    vocab_path = Path(config.vocab_path)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    save_vocab(vocabulary, config.vocab_path)
    logger.info(f"Vocabulary saved to: {config.vocab_path}")
    
    logger.info("Creating datasets and dataloaders...")
    train_dataset = ToxicCommentsDataset(train_df, vocabulary, config.max_seq_len, preprocessor)
    val_dataset = ToxicCommentsDataset(val_df, vocabulary, config.max_seq_len, preprocessor)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    model = LSTMClassifier(
        vocab_size=len(vocabulary),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional
    )
    
    # Initialize trainer and start training
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit()