import numpy as np
import pandas as pd
import pytest

from redcea.clustering.cdr3_grouping import build_len_to_group_id, compute_cdr3_len, map_len_to_group_id


def test_compute_cdr3_len_from_strings():
    lengths = compute_cdr3_len(pd.Series(["CASS", "CASSPG", "CA"]))
    assert lengths.tolist() == [4, 6, 2]


def test_compute_cdr3_len_rejects_mixed_objects():
    with pytest.raises(TypeError):
        compute_cdr3_len(pd.Series(["CASS", 7], dtype=object))


def test_build_len_to_group_id_groups_tail_into_previous_bucket():
    mapping = build_len_to_group_id(np.array([10, 10, 11, 11, 12]), min_frac=0.4)
    assert mapping[10] == 0
    assert mapping[11] == 0
    assert mapping[12] == 0


def test_map_len_to_group_id_uses_nearest_for_unknown_lengths():
    mapping = {10: 0, 12: 1}
    gids = map_len_to_group_id(np.array([10, 11, 12, 13]), mapping, unknown_len_policy="nearest")
    assert gids.tolist() == [0, 0, 1, 1]


def test_map_len_to_group_id_can_raise_for_unknown_lengths():
    with pytest.raises(ValueError):
        map_len_to_group_id(np.array([9, 10]), {10: 0}, unknown_len_policy="error")
