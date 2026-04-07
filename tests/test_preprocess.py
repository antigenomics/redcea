import numpy as np
import pandas as pd

from redcea.clustering.preprocess import prepare_data_for_clustering, standardize_data


def test_standardize_data_centers_and_scales_columns():
    data = np.array([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]], dtype=np.float32)

    standardized = standardize_data(data.copy())

    assert np.allclose(standardized.mean(axis=0), [0.0, 0.0], atol=1e-6)
    assert np.allclose(standardized[:, 0].std(ddof=0), 1.0, atol=1e-6)
    assert np.allclose(standardized[:, 1], 0.0, atol=1e-6)


def test_prepare_data_for_clustering_returns_expected_shape():
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [4.0, 3.0, 2.0, 1.0],
            "f3": [1.0, 1.5, 2.0, 2.5],
        }
    )

    reduced = prepare_data_for_clustering(df, n_components=2)

    assert reduced.shape == (4, 2)
    assert np.isfinite(reduced).all()
