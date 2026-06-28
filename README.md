# MLOps Emotion Detection Pipeline — Group 38
**MLOps | PGD AI Program | IIT**

An end-to-end **Emotion Detection** pipeline built with HuggingFace Transformers, Weights & Biases, Docker, and GitHub Actions.

> `mlops-group-project.ipynb` is kept as a **reference notebook only** — it is not part of this project's data or pipeline.  
> The actual training notebook for this project is **`emotion-detection-kaggle.ipynb`**.

---

## Task
**Multi-class text emotion classification** — given a sentence, predict one of 6 emotions:

| ID | Emotion  | Example                                   |
|----|----------|-------------------------------------------|
| 0  | sadness  | "I feel so empty and alone."              |
| 1  | joy      | "Today was the best day of my life!"      |
| 2  | love     | "I love spending time with my family."    |
| 3  | anger    | "This is completely unacceptable!"        |
| 4  | fear     | "I am terrified of what might happen."    |
| 5  | surprise | "I never expected that, wow!"             |

---

## Important Links

| Component | Link |
|---|---|
| GitHub Repository | https://github.com/Sonu-Kumar-IIT/MLOPS_IIT_GROUP_PROJECT_38 |
| Hugging Face Model | _(push after training: `HF_REPO_ID=your-username/mlops-group38-emotion`)_ |
| W&B Dashboard | _(project: `emotion-mlops-group38-project`)_ |
| Docker Image | _(push via Docker Publish workflow)_ |

---

## Dataset

**`dair-ai/emotion`** — English Twitter messages labelled with 6 emotions  
- Train: 16,000 samples | Validation: 2,000 | Test: 2,000  
- Source: https://huggingface.co/datasets/dair-ai/emotion

### Data Cleaning Decisions

| Issue | Action | Reason |
|---|---|---|
| URLs | Removed | Carry no emotion signal |
| @mentions | Removed | Not relevant to emotion content |
| #hashtag symbols | Stripped `#`, kept word | Hashtag words can carry emotion |
| Non-ASCII characters | Removed | Keeps text clean for tokeniser |
| Non-alphanumeric chars | Replaced with space (keep `.!?,`) | Punctuation carries sentiment |
| Multiple whitespace | Collapsed to single space | Normalise for tokeniser |
| Empty rows post-cleaning | Dropped | Prevent training on empty inputs |

---

## Model

**Base model**: `distilbert-base-uncased`  
- ~66M parameters — 40% smaller than BERT-base, ~97% of performance  
- Uncased is fine for emotion (capitalisation is less critical than in NER)  
- Fits comfortably within Kaggle free GPU tier

---

## Project Structure

```
MLOPS_IIT_GROUP_PROJECT_38/
├── .github/workflows/
│   ├── ci.yml                        # Lint + import check
│   ├── inference.yml                 # Manual emotion inference trigger
│   └── docker-publish.yml            # Build & push Docker image
├── artifacts/
│   ├── id2label.json                 # {0: sadness, 1: joy, ...}
│   └── eval_results/
│       ├── classification_report.json
│       ├── classification_report.txt
│       └── confusion_matrix.png      # Generated during training
├── src/
│   ├── config.py                     # All settings — env-var driven
│   ├── utils.py                      # Logging helpers
│   ├── wandb_utils.py                # W&B init/finish
│   ├── inference/
│   │   ├── infrence_from_hub.py      # Inference via HF Hub
│   │   └── infrence_local.py         # Inference from local checkpoint
│   └── training/
│       ├── data.py                   # Load, clean, tokenise dair-ai/emotion
│       ├── train.py                  # HuggingFace Trainer (SequenceClassification)
│       ├── eval.py                   # Metrics + confusion matrix
│       ├── push_to_hub.py            # Push to HF Hub
│       └── main.py                   # Full pipeline orchestrator
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt                  # Production deps
├── requirements-dev.txt              # Training deps (Kaggle / local)
├── emotion-detection-kaggle.ipynb    # NEW: Kaggle training notebook
└── mlops-group-project.ipynb         # Reference only (not used in pipeline)
```

---

## Training Experiments

| Hyperparameter | V1 (run-v1) | V2 (run-v2) |
|---|---|---|
| Learning rate | 2e-5 | 5e-5 |
| Epochs | 4 | 6 |
| Batch size | 32 | 32 |
| Weight decay | 0.01 | 0.01 |
| Warmup ratio | 0.06 | 0.06 |

---

## Quick Start

### 1. Install training dependencies
```bash
pip install -r requirements-dev.txt
```

### 2. Set environment variables
```bash
export HF_TOKEN="<your-hf-token>"
export WANDB_API_KEY="<your-wandb-key>"
export HF_REPO_ID="<your-username>/mlops-group38-emotion"
```

### 3. Run full pipeline
```bash
python -m src.training.main
```

### 4. Run inference from Hub
```bash
INPUT_TEXT="I feel so happy today!" python -m src.inference.infrence_from_hub
```

### 5. Docker
```bash
docker build --build-arg HF_MODEL_NAME=<your-username>/mlops-group38-emotion -t mlops-group38-emotion .
docker run --rm -e HF_TOKEN="<token>" -e INPUT_TEXT="I am so angry right now!" mlops-group38-emotion
```

---

## GitHub Secrets Required

| Secret | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face API token |
| `WANDB_API_KEY` | W&B experiment tracking |
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `HF_REPO_ID` | HF repo id for the model |

---

## License

MIT — see [LICENSE](LICENSE).
