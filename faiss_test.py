import numpy as np
import hnswlib
import faiss
import time
import os
import psutil

# ============ Настройки ============
N_BG = 200_000      # размер background
N_Q = 20_000        # число запросов (sample)
D = 256             # размерность эмбеддингов
K = 10              # число соседей
THREADS = [1, 4, 8, 16, 32]   # проверим масштабирование
M = 32              # параметр HNSW (глубина связей)
EF_C = 300          # ef_construction
EF_S = 200          # ef_search
# ==================================

def log_memory():
    mem = psutil.virtual_memory()
    print(f"🧠 Memory usage: {mem.used/1e9:.2f} GB / {mem.total/1e9:.2f} GB")

print(f"Creating synthetic data: {N_BG:,} base, {N_Q:,} queries, dim={D}")
bg = np.random.rand(N_BG, D).astype('float32')
sample = np.random.rand(N_Q, D).astype('float32')
log_memory()

# ================================================================
# 1️⃣ FAISS точный поиск (1 поток)
# ================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
faiss.omp_set_num_threads(1)

index_faiss = faiss.IndexFlatL2(D)
index_faiss.add(bg)

t0 = time.time()
D_faiss, I_faiss = index_faiss.search(sample, K)
print(f"\n🧩 FAISS (1 thread): {time.time()-t0:.2f} s")

# ================================================================
# 2️⃣ HNSWlib многопоточный поиск
# ================================================================
index_hnsw = hnswlib.Index(space='l2', dim=D)
index_hnsw.init_index(max_elements=N_BG, ef_construction=EF_C, M=M, random_seed=42)

print("\nBuilding HNSW index...")
t0 = time.time()
index_hnsw.add_items(bg, num_threads=8)
print(f"✅ HNSW build time: {time.time()-t0:.2f} s")

index_hnsw.set_ef(EF_S)

for n_threads in THREADS:
    t0 = time.time()
    _, _ = index_hnsw.knn_query(sample, k=K, num_threads=n_threads)
    dt = time.time() - t0
    print(f"HNSWlib: {n_threads:2d} threads → {dt:6.2f} s")

# ================================================================
# 3️⃣ Проверка качества (recall на 100 запросах)
# ================================================================
subset = np.arange(100)
D_true, I_true = index_faiss.search(sample[subset], K)

# HNSWlib возвращает (labels, distances)
I_h, D_h = index_hnsw.knn_query(sample[subset], k=K, num_threads=8)

recall = np.mean([
    len(set(I_true[i]) & set(I_h[i])) / K for i in range(len(subset))
])
print(f"\n🎯 Recall@{K} (100 queries): {recall*100:.1f}%")
