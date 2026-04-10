from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from umap import UMAP
except ImportError:  # pragma: no cover
    UMAP = None


@dataclass
class BackgroundTransform:
    chain: str
    n_pca_components: int = 50
    umap_n_components: int = 2
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"
    random_state: int = 42
    embedding_columns: list[str] | None = None

    scaler: StandardScaler | None = field(default=None, init=False, repr=False)
    pca: PCA | None = field(default=None, init=False, repr=False)
    umap_model: Any | None = field(default=None, init=False, repr=False)

    background_pca_: np.ndarray | None = field(default=None, init=False, repr=False)
    background_umap_: np.ndarray | None = field(default=None, init=False, repr=False)

    def _as_frame(self, embeddings: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(embeddings, pd.DataFrame):
            return embeddings.copy()
        if self.embedding_columns is None:
            raise ValueError("embedding_columns are not set; cannot transform numpy array input")
        return pd.DataFrame(embeddings, columns=self.embedding_columns)

    def _select_columns(self, embeddings: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        df = self._as_frame(embeddings)
        if self.embedding_columns is None:
            self.embedding_columns = list(df.columns)
            return df

        missing = [c for c in self.embedding_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Input embeddings are missing columns required by BackgroundTransform: {missing}")
        return df.loc[:, self.embedding_columns]

    @property
    def scaler_mean_(self) -> np.ndarray:
        self._require_fitted_scaler()
        return self.scaler.mean_

    @property
    def scaler_scale_(self) -> np.ndarray:
        self._require_fitted_scaler()
        return self.scaler.scale_

    @property
    def pca_components_(self) -> np.ndarray:
        self._require_fitted_pca()
        return self.pca.components_

    @property
    def pca_mean_(self) -> np.ndarray:
        self._require_fitted_pca()
        return self.pca.mean_

    @property
    def pca_explained_variance_(self) -> np.ndarray:
        self._require_fitted_pca()
        return self.pca.explained_variance_

    @property
    def pca_explained_variance_ratio_(self) -> np.ndarray:
        self._require_fitted_pca()
        return self.pca.explained_variance_ratio_

    def _require_fitted_scaler(self) -> None:
        if self.scaler is None:
            raise RuntimeError("BackgroundTransform scaler is not fitted")

    def _require_fitted_pca(self) -> None:
        if self.pca is None:
            raise RuntimeError("BackgroundTransform PCA is not fitted")

    def _require_fitted_umap(self) -> None:
        if self.umap_model is None:
            raise RuntimeError("BackgroundTransform UMAP is not fitted")

    def fit(self, background_embeddings: pd.DataFrame | np.ndarray) -> "BackgroundTransform":
        bg = self._select_columns(background_embeddings)
        x = bg.to_numpy(dtype=np.float32, copy=True)

        self.scaler = StandardScaler(copy=True)
        x_scaled = self.scaler.fit_transform(x).astype(np.float32, copy=False)

        n_components = min(self.n_pca_components, x_scaled.shape[0], x_scaled.shape[1])
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        self.background_pca_ = self.pca.fit_transform(x_scaled).astype(np.float32, copy=False)
        return self

    def transform_pca(self, embeddings: pd.DataFrame | np.ndarray) -> np.ndarray:
        self._require_fitted_scaler()
        self._require_fitted_pca()
        df = self._select_columns(embeddings)
        x = df.to_numpy(dtype=np.float32, copy=True)
        x_scaled = self.scaler.transform(x).astype(np.float32, copy=False)
        return self.pca.transform(x_scaled).astype(np.float32, copy=False)

    def fit_transform_pca(self, background_embeddings: pd.DataFrame | np.ndarray) -> np.ndarray:
        self.fit(background_embeddings)
        return self.background_pca_

    def fit_umap(self, background_embeddings_pca: np.ndarray | None = None) -> np.ndarray:
        if UMAP is None:
            raise ImportError("umap-learn is not installed")
        if background_embeddings_pca is None:
            if self.background_pca_ is None:
                raise RuntimeError("Background PCA coordinates are not available")
            background_embeddings_pca = self.background_pca_

        self.umap_model = UMAP(
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric=self.umap_metric,
            random_state=self.random_state,
            transform_seed=self.random_state,
        )
        self.background_umap_ = self.umap_model.fit_transform(background_embeddings_pca).astype(np.float32, copy=False)
        return self.background_umap_

    def transform_umap(self, embeddings_pca: np.ndarray) -> np.ndarray:
        self._require_fitted_umap()
        return self.umap_model.transform(embeddings_pca).astype(np.float32, copy=False)

    def fit_full(self, background_embeddings: pd.DataFrame | np.ndarray, fit_umap: bool = False) -> "BackgroundTransform":
        self.fit(background_embeddings)
        if fit_umap:
            self.fit_umap()
        return self

    def transform(
        self,
        embeddings: pd.DataFrame | np.ndarray,
        with_umap: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        embeddings_pca = self.transform_pca(embeddings)
        if not with_umap:
            return embeddings_pca
        embeddings_umap = self.transform_umap(embeddings_pca)
        return embeddings_pca, embeddings_umap

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BackgroundTransform":
        obj = load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Object at {path} is not a BackgroundTransform")
        return obj

    def metadata(self) -> dict[str, Any]:
        data = {
            "chain": self.chain,
            "embedding_columns": self.embedding_columns,
            "n_pca_components": None if self.pca is None else int(self.pca.n_components_),
            "random_state": self.random_state,
            "has_umap": self.umap_model is not None,
            "umap_n_components": self.umap_n_components,
            "umap_n_neighbors": self.umap_n_neighbors,
            "umap_min_dist": self.umap_min_dist,
            "umap_metric": self.umap_metric,
        }
        if self.pca is not None:
            data["explained_variance_ratio_sum"] = float(np.sum(self.pca.explained_variance_ratio_))
        if self.background_pca_ is not None:
            data["background_size"] = int(self.background_pca_.shape[0])
        return data
