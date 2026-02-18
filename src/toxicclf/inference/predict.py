import torch
import json
from pathlib import Path

from toxicclf.config import Config
from toxicclf.constants import LABEL_COLUMNS
from toxicclf.data.preprocessing import Preprocessor
from toxicclf.models.lstm import LSTMClassifier
from toxicclf.utils.io import load_vocab
from toxicclf.utils.logger import get_logger
logger = get_logger(__name__)

def load_vocab(path):
    with open(path, "r") as f:
        return json.load(f)


def run_inference(text: str):
    config = Config()
    device = torch.device(config.device)

    artifacts_dir = Path(config.artifacts_dir)

    # Load vocab
    word2idx = load_vocab(config.vocab_path)

    # Reconstruct model
    model = LSTMClassifier(
        vocab_size=len(word2idx),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
    )

    checkpoint = torch.load(
        config.save_checkpoint_dir + f"/{config.saved_model_name}",
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Preprocess input
    preprocessor = Preprocessor()
    tokens = preprocessor.preprocess(text)

    # Encode
    encoded = [
        word2idx.get(token, word2idx.get("<UNK>", 1))
        for token in tokens
    ]

    encoded = encoded[: config.max_seq_len]
    padded = encoded + [0] * (config.max_seq_len - len(encoded))

    input_tensor = torch.tensor(
        [padded],
        dtype=torch.long,
    ).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int().cpu().numpy()[0]

    results = {
        label: int(pred)
        for label, pred in zip(LABEL_COLUMNS, preds)
    }

    logger.info("Predictions:")
    for label, value in results.items():
        logger.info(f"{label}: {value}")
