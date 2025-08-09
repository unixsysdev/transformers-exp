# Transformers-Exp: Williams-Style Space–Time Optimizations

This repository contains experimental implementations of **memory-efficient Transformer attention** mechanisms inspired by Ryan Williams’ space–time tradeoff results.  
The goal: explore **very long sequence processing** on limited hardware by recomputing and chunking attention in mathematically optimal ways.

---

## ✨ Features

- **Multiple KV Storage Policies**
  - `cache` – Standard key/value caching (fastest, highest memory use)
  - `grouped` – Grouped KV heads (memory ↓, similar speed)
  - `recompute` – Williams-style recomputation with chunked queries/keys (memory ↓↓, speed ↓)
- **Chunked Attention**
  - Control **query chunks** (`--q-chunk`) and **key chunks** (`--k-chunk`)
  - Reduces peak activation size from `O(T²)` to `O(T * chunk)`
- **4-bit Projection Quantization**
  - Pack projection weights into int4, with runtime dequantization
- **Flexible Benchmarking**
  - CPU & GPU runs
  - Runtime and peak memory measurement
  - CSV output for analysis
- **Jupyter Visualization**
  - Plot **time vs memory vs sequence length**
  - Compare all policies interactively

---

## 📂 Repository Structure

| File | Purpose |
|------|---------|
| `enhanced_transformer.py` | Core Transformer block with all optimization knobs |
| `bench.py` | CLI benchmarking tool for speed & memory comparisons |
| `williams.ipynb` / `transformer.ipynb` | Experiment notebooks |
| `requirements.txt` | Python dependencies |

---

## 🚀 Quick Start

### 1️⃣ Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Run a benchmark
```bash
python bench.py --seq 1024 4096 --runs 3 --print
```

Example output:
```
seq,policy,k_chunk,q_chunk,kv_heads,time_s,peak_cuda_MB,delta_rss_MB,out_shape
1024,cache,,,,0.0335,,-0.61,(1, 1024, 256)
1024,grouped,,,2,0.0338,,0.34,(1, 1024, 256)
1024,recompute,256,64,,0.2051,,1.69,(1, 1024, 256)
```

---

## 📊 Jupyter Visualization

1. Launch Jupyter:
```bash
jupyter notebook
```
2. Open `williams.ipynb` (or your plotting notebook)  
3. Run all cells to generate time & memory plots

---

## ⚙️ Key CLI Options (bench.py)

| Option | Description |
|--------|-------------|
| `--seq` | Sequence lengths (space-separated) |
| `--runs` | Number of runs per config |
| `--policy` | `cache`, `grouped`, `recompute` |
| `--k-chunk` | Key chunk size (for recompute) |
| `--q-chunk` | Query chunk size (for recompute) |
| `--kv-heads` | Number of KV heads (for grouped) |
| `--print` | Pretty-print results |

---

## 📌 Use Cases

- Testing **space–time tradeoffs** in Transformer attention
- Running **long context inference** (8K–32K tokens) without GPU OOM
- Exploring **edge deployment** of attention models with minimal RAM
- Academic research into **subquadratic attention approximations**

---

## 📝 Notes & Caveats

- **Recompute** is slower for small sequences; shines for 8K+ tokens
- CPU RSS measurements can be noisy; GPU peak memory is more reliable
- Int4 projection quantization is currently **weights-only**
- Not production-optimized; intended for **research and benchmarking**

---

## 📄 License
MIT License – free to use, modify, and distribute.

---

## 🙌 Acknowledgements
Inspired by:
- Ryan Williams, *"A new algorithm for long sequences with less memory"* (space–time tradeoffs)
- FlashAttention & PyTorch SDPA kernels


