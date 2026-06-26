"""
RAG retrieval experiments for Day 3 contract clause search.

Sweeps chunk_size, overlap, and k; records cosine similarities; charts the results.
Reuses the exact chunk/embed/index/search logic from the notebook so results transfer.
"""
import os, glob, time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve paths relative to this script so it runs on any machine / from any cwd.
DAY03 = os.path.dirname(os.path.abspath(__file__))
CONTRACTS_DIR = os.path.join(DAY03, "contracts")
OUT_DIR = DAY03  # save charts next to the notebook

QUERIES = [
    "non-compete period longer than one year",
    "restrictions on hiring our employees after the contract ends",
    "who owns the IP if we co-develop something",
    "limitation of liability and indemnification",
    "obligation to keep information confidential",
]

# ---------- notebook logic (identical) ----------
def strip_header(raw):
    if "=" * 10 in raw:
        return raw.split("=" * 10, 1)[1].lstrip("=").lstrip()
    return raw

def load_documents(folder_path):
    docs = []
    for path in sorted(glob.glob(os.path.join(folder_path, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            text = strip_header(f.read())
        docs.append({"text": text, "source": os.path.basename(path)})
    return docs

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks, start = [], 0
    step = max(1, chunk_size - overlap)  # guard against overlap >= chunk_size
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += step
    return chunks

def build_chunks(documents, chunk_size, overlap):
    all_chunks = []
    for doc in documents:
        for i, piece in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            all_chunks.append({"text": piece, "source": doc["source"], "chunk_id": i})
    return all_chunks

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    vecs = embeddings.copy()
    faiss.normalize_L2(vecs)
    index.add(vecs)
    return index

def search_scores(index, embedder, query, k):
    q = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, k)
    return scores[0], idxs[0]

# ---------- run ----------
print("Loading embedding model (first run may download ~80MB)...")
t0 = time.time()
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print(f"  model ready in {time.time()-t0:.1f}s")

documents = load_documents(CONTRACTS_DIR)
print(f"Loaded {len(documents)} contracts")

def run_config(chunk_size, overlap, k=5):
    """Return (n_chunks, mean_top1, mean_topk_scores[list len k])."""
    chunks = build_chunks(documents, chunk_size, overlap)
    texts = [c["text"] for c in chunks]
    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    index = build_index(emb)
    per_query_topk = []
    for q in QUERIES:
        scores, _ = search_scores(index, embedder, q, k)
        per_query_topk.append(scores)
    arr = np.array(per_query_topk)  # (n_queries, k)
    return len(chunks), float(arr[:, 0].mean()), arr.mean(axis=0)

# Experiment A: chunk_size sweep (overlap fixed at 50)
CHUNK_SIZES = [100, 250, 500, 1000, 2000]
FIXED_OVERLAP = 50
print("\n=== Experiment A: chunk size sweep (overlap=50) ===")
A_nchunks, A_top1 = [], []
for cs in CHUNK_SIZES:
    n, top1, _ = run_config(cs, FIXED_OVERLAP, k=5)
    A_nchunks.append(n); A_top1.append(top1)
    print(f"  chunk_size={cs:5d}  chunks={n:4d}  mean_top1_cosine={top1:.4f}")

# Experiment B: overlap sweep (chunk_size fixed at 500)
OVERLAPS = [0, 25, 50, 100, 200, 400]
FIXED_CHUNK = 500
print("\n=== Experiment B: overlap sweep (chunk_size=500) ===")
B_nchunks, B_top1 = [], []
for ov in OVERLAPS:
    n, top1, _ = run_config(FIXED_CHUNK, ov, k=5)
    B_nchunks.append(n); B_top1.append(top1)
    print(f"  overlap={ov:4d}  chunks={n:4d}  mean_top1_cosine={top1:.4f}")

# Experiment C: k sweep / score-by-rank (chunk_size=500, overlap=50)
K_MAX = 10
print("\n=== Experiment C: score by rank position k (chunk_size=500, overlap=50) ===")
_, _, C_scores_by_rank = run_config(FIXED_CHUNK, 50, k=K_MAX)
for rank, s in enumerate(C_scores_by_rank, 1):
    print(f"  rank {rank:2d}  mean_cosine={s:.4f}")

# Experiment D: 2D grid chunk_size x overlap -> mean top1 (heatmap)
GRID_CHUNKS = [100, 250, 500, 1000]
GRID_OVERLAPS = [0, 50, 100, 200]
print("\n=== Experiment D: chunk_size x overlap grid (mean top1 cosine) ===")
D = np.zeros((len(GRID_CHUNKS), len(GRID_OVERLAPS)))
for i, cs in enumerate(GRID_CHUNKS):
    for j, ov in enumerate(GRID_OVERLAPS):
        ov_eff = min(ov, cs - 10)  # keep overlap < chunk_size
        _, top1, _ = run_config(cs, ov_eff, k=5)
        D[i, j] = top1
    print(f"  chunk_size={cs}: " + "  ".join(f"ov{GRID_OVERLAPS[j]}={D[i,j]:.3f}" for j in range(len(GRID_OVERLAPS))))

# ---------- charts ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Day 3 RAG retrieval experiments — contract clause search\n(mean over 5 queries, all-MiniLM-L6-v2, FAISS IndexFlatIP)", fontsize=13, fontweight="bold")

# A: chunk size
ax = axes[0, 0]
ax.plot(CHUNK_SIZES, A_top1, "o-", color="#1f77b4", lw=2, label="mean top-1 cosine")
ax.set_xlabel("chunk_size (words)"); ax.set_ylabel("mean top-1 cosine", color="#1f77b4")
ax.set_title("A. Effect of chunk size (overlap=50)")
ax.tick_params(axis="y", labelcolor="#1f77b4"); ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(CHUNK_SIZES, A_nchunks, "s--", color="#ff7f0e", alpha=0.7, label="# chunks")
ax2.set_ylabel("# chunks", color="#ff7f0e"); ax2.tick_params(axis="y", labelcolor="#ff7f0e")

# B: overlap
ax = axes[0, 1]
ax.plot(OVERLAPS, B_top1, "o-", color="#1f77b4", lw=2)
ax.set_xlabel("overlap (words)"); ax.set_ylabel("mean top-1 cosine", color="#1f77b4")
ax.set_title("B. Effect of overlap (chunk_size=500)")
ax.tick_params(axis="y", labelcolor="#1f77b4"); ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(OVERLAPS, B_nchunks, "s--", color="#ff7f0e", alpha=0.7)
ax2.set_ylabel("# chunks", color="#ff7f0e"); ax2.tick_params(axis="y", labelcolor="#ff7f0e")

# C: score by rank (k)
ax = axes[1, 0]
ranks = list(range(1, K_MAX + 1))
ax.plot(ranks, C_scores_by_rank, "o-", color="#2ca02c", lw=2)
ax.set_xlabel("rank position (k)"); ax.set_ylabel("mean cosine at that rank")
ax.set_title("C. Score by rank — why bigger k ≠ higher score")
ax.set_xticks(ranks); ax.grid(alpha=0.3)
ax.annotate("k only extends the list;\neach score is fixed", xy=(6, C_scores_by_rank[5]),
            xytext=(5.5, C_scores_by_rank[0]*0.85),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9, color="gray")

# D: heatmap
ax = axes[1, 1]
im = ax.imshow(D, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(GRID_OVERLAPS))); ax.set_xticklabels(GRID_OVERLAPS)
ax.set_yticks(range(len(GRID_CHUNKS))); ax.set_yticklabels(GRID_CHUNKS)
ax.set_xlabel("overlap (words)"); ax.set_ylabel("chunk_size (words)")
ax.set_title("D. mean top-1 cosine across chunk_size × overlap")
for i in range(len(GRID_CHUNKS)):
    for j in range(len(GRID_OVERLAPS)):
        ax.text(j, i, f"{D[i,j]:.3f}", ha="center", va="center",
                color="white" if D[i, j] < D.mean() else "black", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_png = os.path.join(OUT_DIR, "rag_experiments_chart.png")
plt.savefig(out_png, dpi=130)
print(f"\nChart saved to: {out_png}")
