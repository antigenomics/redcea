"""
tcrempnet_utils.py
Auto-generated from 2_analysis.ipynb.

This module consolidates reusable functions/classes detected in your notebook.
If you see NOTE blocks about hardcoded paths, consider refactoring those
into function parameters.
"""
# flake8: noqa
# pylint: skip-file

from collections import defaultdict
from mir.basic.pgen import OlgaModel
from mir.basic.sampling import RepertoireSampling
from mir.basic.segment_usage import *
from mir.biomarkers.fisher_biomarkers_detector import FisherBiomarkersDetector
from mir.common.clonotype import ClonotypeAA
from mir.common.clonotype_dataset import ClonotypeDataset
from mir.common.parser import *
from mir.common.repertoire import Repertoire
from mir.common.repertoire_dataset import RepertoireDataset
from mir.comparative.pair_matcher import ClonotypeRepresentation
from mir.comparative.pair_matcher import PairMatcher
from pympler.asizeof import asizeof
from scipy.spatial import KDTree
from scipy.stats import fisher_exact
from sklearn.preprocessing import MinMaxScaler
import logomaker
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import time
from collections import defaultdict
import networkx as nx
import pandas as pd
import numpy as np
import pandas as pd
from multiprocessing import Pool

def _pgen_worker(seq: str):
    try:
        from mir.basic.pgen import OlgaModel
        olga = OlgaModel()
        return float(olga.compute_pgen_cdr3aa(seq))
    except Exception:
        return np.nan

def compute_pgen_pool(df: pd.DataFrame,
                      seq_col: str = "cdr3aa_beta",
                      out_col: str = "pgen",
                      processes: int = 32,
                      chunksize: int = 500,
                      progress: bool = True) -> pd.Series:
    """Рассчитать pgen для df[seq_col] в несколько процессов."""
    seqs = df[seq_col].astype(str).tolist()

    iterator = seqs
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(seqs, desc="pgen", mininterval=0.2)
        except Exception:
            pass

    with Pool(processes=processes) as pool:
        vals = list(pool.imap(_pgen_worker, iterator, chunksize=chunksize))

    return pd.Series(vals, index=df.index, name=out_col)


def merge_clusters_by_shared_cdr3(df,
                                  cluster_col: str = "cluster_id",
                                  cdr3_col: str = "cdr3aa_beta",
                                  merged_col: str = "merged_cluster_id"):
    """
    Объединяет кластеры, которые разделяют хотя бы один одинаковый CDR3.
    Реализация соответствует исходному коду на ячейках.

    Параметры
    ---------
    df : pd.DataFrame
        Таблица с колонками [cluster_col, cdr3_col].
    cluster_col : str, по умолчанию "cluster_id"
        Колонка с ID кластеров.
    cdr3_col : str, по умолчанию "cdr3aa_beta"
        Колонка с CDR3-амино кислотными последовательностями.
    merged_col : str, по умолчанию "merged_cluster_id"
        Имя новой колонки для объединённых кластеров.

    Возвращает
    ----------
    df_out : pd.DataFrame
        Копия входного датафрейма с добавленной колонкой [merged_col].
    mapping : dict
        Словарь cluster_id -> merged_cluster_id.
    """
    cdr3_to_clusters = defaultdict(set)
    for _, row in df.iterrows():
        cdr3_to_clusters[row[cdr3_col]].add(row[cluster_col])

    G = nx.Graph()
    G.add_nodes_from(df[cluster_col].unique())
    for clusters in cdr3_to_clusters.values():
        clusters = list(clusters)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                G.add_edge(clusters[i], clusters[j])

    cluster_mapping = {}
    for new_id, component in enumerate(nx.connected_components(G)):
        for cid in component:
            cluster_mapping[cid] = new_id

    df_out = df.copy()
    df_out[merged_col] = df_out[cluster_col].map(cluster_mapping)
    return df_out, cluster_mapping

def get_sample_info(sample_name, airr_path='/projects/immunestatus/emerson/airr_format', 
                    tcrempnet_path = f'/projects/immunestatus/emerson_ebv/tcrempnet'):
    """
    NOTE: hardcoded paths detected. Consider parameterizing:
    - /projects/immunestatus/emerson/airr_format/{sample_name}.tsv
    """
    summary = pd.read_csv(f'{tcrempnet_path}/{sample_name}_summary_tcrempnet.tsv', sep='\t')
    c15 = len(pd.read_csv(
        f'{airr_path}/{sample_name}.tsv', sep='\t'))
    print(f'''
    sample: {sample_name}
    clusters: {len(summary)}
    overall from sample: {c15}
    ''')
    return summary, c15

def get_clonotypes(sample_name, data_path='/projects/immunestatus/emerson_ebv'):
    data_path = f'{data_path}/tcrempnet'
    df = pd.read_csv(f'{data_path}/{sample_name}_enriched_clonotypes_tcremp_pgen.tsv', sep='\t')
    return df

def has_vdjdb_match(x, vdjdb, threshold=1):
    return len(vdjdb.get_matching_clonotypes(x, threshold=threshold))

def get_cluster_matches_vdjdb(df, cluster_id, vdjdb):
    df = df[df.cluster_id == cluster_id]
    matches = df.cdr3aa_beta.apply(lambda x: has_vdjdb_match(x, vdjdb))
    return sum(matches) > 0

def plot_volcano(df,
                 pval_threshold=0.05,
                 fold_threshold=0,
                 sample_name="",
                 ax=None,
                 layers=None,
                 layer_exclusive=True,
                 combo_style=None,
                 point_size=None):
    """
    Volcano plot с возможностью отображения нескольких слоёв точек и
    отдельным стилем для точек, у которых несколько маркеров True.
    В заголовке дополнительно показываются количества кластеров,
    матчущихся на каждый из переданных layers.
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = df.copy()
    eps = 1e-10
    df['log10_pval'] = -np.log10(df['enrichment_pvalue_zbinom'] + eps)
    df['significant'] = (
        (df['enrichment_fdr_zbinom'] < pval_threshold) &
        (df['log_fold_change'] > fold_threshold)
    )

    # --------- Счётчики по слоям (до любых фильтраций combo и т.п.) ----------
    layer_counts = {}
    if layers:
        for layer in layers:
            col = layer.get("col")
            if col in df.columns:
                label = layer.get("label", col)
                layer_counts[label] = int(df[col].fillna(False).sum())

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    main_color = '#FFCB9A'
    base_color = '#D2E8E3'

    # стили для комбо по умолчанию
    if combo_style is None:
        combo_style = {"marker": "D", "color": "purple", "edgecolor": "black",
                       "linewidth": 0.4, "alpha": 1.0, "label": "combo"}

    # вычисляем маски для слоёв
    layer_cols = [l["col"] for l in (layers or []) if l.get("col") in df.columns]
    layer_any = np.zeros(len(df), dtype=bool)
    if layer_cols:
        for col in layer_cols:
            layer_any |= df[col].fillna(False).values

    if layer_exclusive:
        base_mask = ~layer_any
    else:
        base_mask = np.ones(len(df), dtype=bool)

    # базовый слой
    base_df = df.loc[base_mask]
    if len(base_df):
        sns.scatterplot(
            data=base_df,
            x="log_fold_change",
            y="log10_pval",
            hue="significant",
            palette={True: main_color, False: base_color},
            edgecolor="black",
            linewidth=0.3,
            alpha=0.85,
            ax=ax,
            **({"s": point_size} if point_size else {})
        )

    # комбо: точки у которых >1 маркер True
    if len(layer_cols) >= 2:
        combo_mask = df[layer_cols].sum(axis=1) > 1
        combo_df = df.loc[combo_mask]
        if len(combo_df):
            sns.scatterplot(
                data=combo_df,
                x="log_fold_change",
                y="log10_pval",
                marker=combo_style["marker"],
                color=combo_style["color"],
                edgecolor=combo_style["edgecolor"],
                linewidth=combo_style["linewidth"],
                alpha=combo_style["alpha"],
                ax=ax,
                label=combo_style["label"],
                **({"s": point_size} if point_size else {})
            )
        # исключаем их из индивидуальных слоёв (как и раньше)
        df = df.loc[~combo_mask]

    # индивидуальные слои
    if layers:
        for layer in layers:
            col = layer["col"]
            if col not in df.columns:
                continue
            m = df[col].fillna(False).astype(bool).values
            if not m.any():
                continue
            sns.scatterplot(
                data=df.loc[m],
                x="log_fold_change",
                y="log10_pval",
                marker=layer.get("marker", "o"),
                color=layer.get("color", "red"),
                edgecolor=layer.get("edgecolor", "black"),
                linewidth=layer.get("linewidth", 0.4),
                alpha=layer.get("alpha", 1.0),
                ax=ax,
                label=layer.get("label", col),
                **({"s": point_size} if point_size else {})
            )

    # пороговые линии
    sig_mask = df['enrichment_fdr_zbinom'] < pval_threshold
    bonferroni_thresh = float(df.loc[sig_mask, 'enrichment_pvalue_zbinom'].max()) if sig_mask.any() else 1.0
    log10_bonferroni = -np.log10(bonferroni_thresh + eps)

    ax.axvline(fold_threshold, ls="--", color="black")
    ax.axhline(-np.log10(pval_threshold + eps), ls="--", color="black")
    ax.axhline(log10_bonferroni, ls=":", color="blue")

    num_significant = int(df["significant"].sum())
    # --------- Формирование заголовка с добавлением layer-counts ----------
    if layer_counts:
        layer_part = " | " + "; ".join([f"{k}: {v}" for k, v in layer_counts.items()])
    else:
        layer_part = ""
    ax.set_title(
        f"{sample_name}: # clusters (fdr < {pval_threshold}): {num_significant}\n{layer_part}"
    )

    ax.set_xlabel("log2(Fold Enrichment)")
    ax.set_ylabel("-log10(p-value)")
    ax.legend()

    return ax


def plot_logo(clonotypes):
    mat_df = logomaker.alignment_to_matrix(clonotypes)
    logomaker.Logo(mat_df, color_scheme='skylign_protein', ax=None)

def get_cluster_usage(df,
                      cluster_idx,
                      clonotype_to_patients,
                      merged_col: str = "merged_cluster_id",
                      print_info: bool = False) -> int:
    all_patients = set()
    for clono_idx in df[df[merged_col] == cluster_idx].index:
        patients = clonotype_to_patients[clono_idx]
        all_patients |= patients
    return len(all_patients)

def compute_topsis_score(df: pd.DataFrame, metric_types: dict, weight_dict: dict = None):
    df = df.copy()
    metrics = list(metric_types.keys())

    # Шаг 1: Нормализация
    norm_df = df[metrics].copy()
    scaler = MinMaxScaler()
    norm_df[metrics] = scaler.fit_transform(norm_df[metrics])

    # Шаг 2: Учет направлений метрик
    for col, kind in metric_types.items():
        if kind == 'cost':
            norm_df[col] = 1 - norm_df[col]

    # Шаг 3: Применение весов (по умолчанию — равные)
    if weight_dict is None:
        weights = np.ones(len(metrics)) / len(metrics)
    else:
        weights = np.array([weight_dict[m] for m in metrics])
        weights = weights / weights.sum()  # нормализация весов
    weighted = norm_df * weights

    # Шаг 4: Расчет расстояний до идеала и анти-идеала
    ideal = weighted.max()
    anti_ideal = weighted.min()

    d_pos = np.linalg.norm(weighted - ideal, axis=1)
    d_neg = np.linalg.norm(weighted - anti_ideal, axis=1)

    # Шаг 5: Финальный TOPSIS скор
    score = d_neg / (d_pos + d_neg)
    # score = (score - score.min()) / (score.max() - score.min()) 
    df["topsis_score"] = score

    return df.sort_values("topsis_score", ascending=False).reset_index(drop=True)