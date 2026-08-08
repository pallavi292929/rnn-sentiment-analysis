# IMDB Sentiment Analysis (RNN, PyTorch)

A sentiment classifier for movie reviews — full NLP preprocessing pipeline, TF-IDF vectorization, and a recurrent neural network trained in PyTorch to predict positive vs. negative sentiment.

> 📊 **[View the interactive project presentation](https://pallavi292929.github.io/rnn-sentiment-analysis/presentation.html)** — pipeline walkthrough, architecture, and results in one page.

**Result: 85.67% test accuracy** on 9,917 held-out IMDB reviews.

---

## Problem Statement

The [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) contains 50,000 movie reviews labeled positive or negative. Raw review text is messy — HTML tags, URLs, punctuation, inconsistent casing, and filler words — so the real work in a sentiment classifier is turning that noise into a clean, structured representation a model can actually learn from. This project builds that full pipeline end to end: clean → vectorize → classify.

## Approach

### 1. Data cleaning
- Loaded 50,000 reviews, dropped duplicates → 49,582 unique reviews.
- Checked for nulls (none found).

### 2. Text preprocessing pipeline
Applied in sequence to every review:

| Step | What it does |
|---|---|
| Lowercase | Normalizes casing |
| Remove URLs | Strips `http...` links via regex |
| Remove punctuation | Keeps only alphanumeric characters and whitespace |
| Remove HTML tags | Strips leftover `<br />`-style tags from the raw scrape |
| Remove stopwords | Tokenizes with NLTK, filters common English stopwords |
| Stemming | Reduces words to root form with NLTK's Porter Stemmer |

### 3. Encoding & vectorization
- Label-encoded the target (`positive` / `negative` → `1` / `0`).
- Vectorized cleaned review text with **TF-IDF** (`max_features=5000`), producing a 5,000-dimensional sparse feature vector per review.

### 4. Modeling
- 80/20 train-test split → 39,665 training reviews, 9,917 test reviews.
- Wrapped in PyTorch `TensorDataset` / `DataLoader` (batch size 64).
- A single-layer **RNN** (hidden size 128) takes each review's TF-IDF vector, followed by a fully-connected layer and sigmoid activation for binary classification.
- Trained with `BCELoss` and the Adam optimizer for 10 epochs.

## Results

| Metric | Value |
|---|---|
| **Test accuracy** | **85.67%** |
| Final training loss (epoch 10) | 0.194 |

Training loss per epoch:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Loss | 0.248 | 0.280 | 0.252 | 0.225 | 0.295 | 0.219 | 0.176 | 0.127 | 0.155 | 0.194 |

## Project Structure

```
.
├── Rnn_sentiment_analysis.ipynb   # Main notebook: preprocessing, TF-IDF, RNN, training, evaluation
├── presentation.html              # Interactive visual walkthrough of the project
└── README.md
```

## Getting Started

### Requirements
```bash
pip install pandas nltk scikit-learn torch
```

On first run, NLTK will download its `punkt` and `stopwords` data automatically via the notebook's `nltk.download(...)` calls.

### Run it
```bash
git clone https://github.com/<your-username>/rnn-sentiment-analysis.git
cd rnn-sentiment-analysis
jupyter notebook Rnn_sentiment_analysis.ipynb
```

> Note: `IMDB Dataset.csv` is not included in this repo (it's a few hundred MB) — download it from [Kaggle's IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in the project root before running.

## Next Steps

- **Fix substring-based stopword removal** — the current implementation removes stopwords with `text.replace(word, "")`, which can strip matching substrings out of *other* words too, not just whole-word matches. Rebuilding the string from filtered tokens (`" ".join(w for w in tokens if w not in stop_words)`) would be safer.
- **Use word embeddings instead of TF-IDF** — feeding a single 5,000-dim TF-IDF vector as a one-timestep "sequence" doesn't let the RNN use its sequential strength. Tokenizing into word sequences with an embedding layer (or pretrained embeddings like GloVe) would let the RNN actually model word order.
- **Try LSTM or GRU cells** — better at capturing longer-range dependencies in review text than a vanilla RNN.
- **Bidirectional RNN** — sentiment often depends on context from both directions in a sentence.
- **Fine-tune a pretrained transformer** (e.g. DistilBERT) for a strong accuracy upgrade with the tradeoff of more compute.

## License

This project is open source under the [MIT License](LICENSE).
