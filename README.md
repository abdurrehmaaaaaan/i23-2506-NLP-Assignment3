# i23-2506-NLP-Assignment3

## Overview

This notebook implements a three-stage NLP pipeline on the Amazon Reviews dataset:

- **Part A** — Encoder-only Transformer for multi-task sentiment and review length classification
- **Part B** — Cosine similarity retrieval module using encoder embeddings
- **Part C** — Decoder-only Transformer for RAG-enhanced explanation generation + ablation study

---

## Requirements

**Python version:** 3.10.11

Install all dependencies with:

```bash
pip install torch numpy matplotlib scikit-learn tqdm
```

All standard library modules used (`json`, `gzip`, `os`, `re`, `math`, `random`, `pickle`, `collections`, `warnings`) are included with Python and require no installation.

---

## Dataset Setup

Download the dataset from: https://nijianmo.github.io/amazon/index.html

You need the following three files in **JSONL format** (one JSON object per line):

| File | Category |
|------|----------|
| `Beauty_5.json` | Beauty |
| `Cell_Phones_and_Accessories_5.json` | Cell Phones & Accessories |
| `Sports_and_Outdoors_5.json` | Sports & Outdoors |

Place all three files inside a folder named `Dataset/` in the same directory as the notebook


The notebook loads 13,000 samples from each category (39,000 total).

---

## Running the Notebook

Open the notebook in Jupyter and run all cells **from top to bottom** in order. Do not skip or re-run cells out of order as later cells depend on variables and saved files from earlier ones.

```bash
jupyter notebook NLP_ASSIGNMENT3.ipynb
```

Or with JupyterLab:

```bash
jupyter lab NLP_ASSIGNMENT3.ipynb
```

The notebook is divided into three sequential parts. Each part must complete fully before the next begins, as Parts B and C load files saved by the previous parts.

---

## Hardware Note

The notebook was developed and run on **CPU** (`Device: cpu`). No GPU is required. Training times on CPU are approximately:

- Part A encoder (3 epochs, 27k samples): ~5–10 minutes
- Part B embedding generation (27k + 5.8k samples): ~2–3 minutes
- Part C decoder (5 epochs): ~10–15 minutes

If a CUDA-capable GPU is available, the notebook will detect and use it automatically via `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.

---

## Hyperparameters at a Glance

| Parameter | Encoder (Part A) | Decoder (Part C) |
|-----------|-----------------|-----------------|
| d_model | 128 | 128 |
| num_heads | 4 | 4 |
| num_layers | 2 | 2 |
| d_ff | 256 | 256 |
| dropout | 0.1 | 0.1 |
| max_seq_len | 128 | 120 (80 src + 40 tgt) |
| batch_size | 64 | 64 |
| learning_rate | 3e-4 | 3e-4 |
| epochs | 3 | 5 |
| optimizer | Adam | Adam |
| scheduler | CosineAnnealing | CosineAnnealing |

Retrieval: `k = 3`, similarity metric: cosine similarity

---

## Troubleshooting

**FileNotFoundError on dataset files** — Make sure the three `.json` files are inside a `Dataset/` folder in the same directory as the notebook, with the exact filenames listed above.

**ModuleNotFoundError** — Run `pip install <package>` for any missing package. The notebook also installs `tqdm` automatically at the start of Part B using `pip` via a system call.

**Out of memory on CPU** — Reduce `BATCH_SIZE` from 64 to 32 in the relevant cells (Part A dataset cell and Part C decoder dataset cell).

**Kernel crashes or long runtimes** — The embedding generation step in Part B encodes all 27,263 training samples and may take several minutes on slower machines. This is expected behaviour.
