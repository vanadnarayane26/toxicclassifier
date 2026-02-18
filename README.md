# Toxic Comment Classifier


A modular, multi-label toxic comment classifier built using PyTorch.

This project implements an end-to-end NLP pipeline including:

- Text preprocessing (NLTK)
- Frequency-based vocabulary building
- Bidirectional LSTM model
- Multi-label classification
- BCEWithLogitsLoss
- Micro and Macro F1 evaluation
- GPU auto-detection
- CLI interface
- Checkpointing & artifact saving
- Clean package structure using Poetry

---

## Problem

Given a comment text, predict whether it belongs to one or more of the following toxicity categories:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

This is a **multi-label classification** problem.

---

## Project Structure
```
toxicclassifier/
│
├── src/toxicclf/
│ ├── data/ # Loading, preprocessing, vocabulary, dataset
│ ├── models/ # LSTM architecture
│ ├── training/ # Trainer + metrics
│ ├── inference/ # Prediction pipeline
│ ├── utils/ # I/O helpers
│ ├── train.py # Training entry logic
│ └── cli.py # CLI interface
│
├── artifacts/ # Saved models & vocabulary
├── tests/
└── pyproject.toml
```


## Installation

Clone the repo
```
git clone <repo-url>
cd toxicclassifier
```

#### Install Dependencies

```
poetry install
```

## Training 

For training using cli, from the command line run the following command:

```
poetry run toxicclf train --data path/to/train.csv
```

Post training, the artifacts will have the vocabulary developed and the saved weights, which can be used while inference.

## Inference

Run the following:

```
poetry run toxicclf predict --text YOUR TEXT
```

## Evaluation

Metrics computed:

Micro F1

Macro F1

**Macro F1 is used for checkpoint selection.**

## Further Enhancements

1. Due to lack of compute, model trained on very few epochs, can be trained for **more epochs**. 

2. Addition of **early stopping**.

3. Use of **static embeddings** like word2ved, Glove etc.

4. Using a **transformer** based model.

5. Configurable **thresholds** for prediction.

6. **FastAPI** deployment and **Docker containerization**.


## Author
**Vanad Narayane**

GitHub: [vanadnarayane26](https://github.com/vanadnarayane26)

LinkedIn: [Vanad Narayane](https://www.linkedin.com/in/vanad-narayane-601936169)

## License
MIT License