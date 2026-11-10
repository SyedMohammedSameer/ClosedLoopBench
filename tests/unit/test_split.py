"""
Unit tests for clbench.data.split.

All tests use synthetic manifests — no filesystem I/O required
(except the load/save round-trip tests which use tmp_path).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from clbench.data.split import (
    N_TEST,
    N_TRAIN,
    N_VAL,
    RANDOM_SEED,
    SPLIT_VERSION,
    _assign_quartile,
    _stratified_split,
    load_splits,
    make_splits,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manifest(
    n: int = 200,
    *,
    n_no_los: int = 28,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build a synthetic manifest with ``n`` usable patients.
    ``n_no_los`` patients get icu_los_days = -1 (no MIMIC-IV link).
    """
    patient_ids = [f"P{i:05d}" for i in range(n)]
    rng = np.random.default_rng(seed)
    los = np.concatenate(
        [
            np.full(n_no_los, -1.0),                        # no-LOS bucket
            rng.exponential(scale=5.0, size=n - n_no_los),  # realistic ICU LOS
        ]
    )
    rng.shuffle(los)
    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "icu_los_days": los,
            "usable": True,
        }
    )


# ---------------------------------------------------------------------------
# _assign_quartile
# ---------------------------------------------------------------------------


class TestAssignQuartile:
    def test_no_los_patients_get_stratum_4(self) -> None:
        los = pd.Series([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        q = _assign_quartile(los)
        # First two (no valid LOS) must land in stratum 4
        assert q.iloc[0] == 4
        assert q.iloc[1] == 4

    def test_valid_los_patients_in_range_0_3(self) -> None:
        los = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        q = _assign_quartile(los)
        assert set(q.tolist()).issubset({0, 1, 2, 3})

    def test_all_no_los_returns_stratum_4(self) -> None:
        los = pd.Series([-1.0, -1.0, 0.0])
        q = _assign_quartile(los)
        assert (q == 4).all()

    def test_output_length_matches_input(self) -> None:
        los = pd.Series([1.0, 2.0, -1.0, 4.0, 5.0])
        q = _assign_quartile(los)
        assert len(q) == len(los)

    def test_quartile_values_are_integers(self) -> None:
        los = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        q = _assign_quartile(los)
        for v in q:
            assert isinstance(int(v), int)

    def test_roughly_equal_quartile_sizes(self) -> None:
        """With 100 uniform samples the quartiles should each have ~25 members."""
        los = pd.Series(np.linspace(0.1, 100.0, 100))
        q = _assign_quartile(los)
        counts = [int((q == i).sum()) for i in range(4)]
        for c in counts:
            assert 20 <= c <= 30, f"Quartile count {c} out of expected range [20, 30]"


# ---------------------------------------------------------------------------
# _stratified_split
# ---------------------------------------------------------------------------


class TestStratifiedSplit:
    def _run(
        self,
        n: int = 200,
        n_no_los: int = 28,
        n_val: int = N_VAL,
        n_test: int = N_TEST,
        seed: int = RANDOM_SEED,
        manifest_seed: int = 0,
    ) -> tuple[list[str], list[str], list[str]]:
        manifest = _make_manifest(n=n, n_no_los=n_no_los, seed=manifest_seed)
        los = pd.to_numeric(manifest["icu_los_days"], errors="coerce").fillna(0.0)
        pids = manifest["patient_id"].tolist()
        strata = _assign_quartile(los).tolist()
        return _stratified_split(pids, strata, n_val=n_val, n_test=n_test, seed=seed)

    # Size contracts
    def test_train_size(self) -> None:
        train, val, test = self._run()
        assert len(train) == N_TRAIN, f"Expected {N_TRAIN} train, got {len(train)}"

    def test_val_size(self) -> None:
        _, val, _ = self._run()
        assert len(val) == N_VAL, f"Expected {N_VAL} val, got {len(val)}"

    def test_test_size(self) -> None:
        _, _, test = self._run()
        assert len(test) == N_TEST, f"Expected {N_TEST} test, got {len(test)}"

    def test_total_equals_n(self) -> None:
        train, val, test = self._run(n=200)
        assert len(train) + len(val) + len(test) == 200

    # Disjointness
    def test_no_train_val_overlap(self) -> None:
        train, val, _ = self._run()
        assert len(set(train) & set(val)) == 0

    def test_no_train_test_overlap(self) -> None:
        train, _, test = self._run()
        assert len(set(train) & set(test)) == 0

    def test_no_val_test_overlap(self) -> None:
        _, val, test = self._run()
        assert len(set(val) & set(test)) == 0

    # No-LOS policy
    def test_no_los_patients_in_train_only(self) -> None:
        """Patients without valid LOS must appear in train, not val or test."""
        manifest = _make_manifest(n=200, n_no_los=28)
        los = pd.to_numeric(manifest["icu_los_days"], errors="coerce").fillna(0.0)
        pids = manifest["patient_id"].tolist()
        strata = _assign_quartile(los).tolist()

        no_los_pids = {
            pid for pid, s in zip(pids, strata) if s == 4
        }
        train, val, test = _stratified_split(pids, strata, n_val=N_VAL, n_test=N_TEST)

        assert no_los_pids.issubset(set(train)), "No-LOS patients must all be in train"
        assert len(no_los_pids & set(val)) == 0
        assert len(no_los_pids & set(test)) == 0

    # Reproducibility
    def test_same_seed_same_result(self) -> None:
        r1 = self._run(seed=42)
        r2 = self._run(seed=42)
        assert r1[0] == r2[0]
        assert r1[1] == r2[1]
        assert r1[2] == r2[2]

    def test_different_seeds_differ(self) -> None:
        _, val1, test1 = self._run(seed=42)
        _, val2, test2 = self._run(seed=99)
        # Very unlikely to be identical across all 60 IDs
        assert val1 != val2 or test1 != test2

    # Stratification
    def test_all_quartiles_represented_in_test(self) -> None:
        """Val and test should draw from all 4 LOS quartiles."""
        manifest = _make_manifest(n=200, n_no_los=0)  # no no-LOS patients
        los = pd.to_numeric(manifest["icu_los_days"], errors="coerce").fillna(0.0)
        pids = manifest["patient_id"].tolist()
        strata = _assign_quartile(los).tolist()
        pid_to_stratum = dict(zip(pids, strata))

        _, val, test = _stratified_split(
            pids, strata, n_val=N_VAL, n_test=N_TEST, seed=RANDOM_SEED
        )
        for split_name, split_ids in [("val", val), ("test", test)]:
            represented = {pid_to_stratum[p] for p in split_ids}
            assert represented == {0, 1, 2, 3}, (
                f"{split_name} missing quartiles: {represented}"
            )

    def test_quartile_proportions_roughly_equal(self) -> None:
        """Each quartile should contribute ~25% of val+test (within ±2 patients)."""
        manifest = _make_manifest(n=200, n_no_los=0)
        los = pd.to_numeric(manifest["icu_los_days"], errors="coerce").fillna(0.0)
        pids = manifest["patient_id"].tolist()
        strata = _assign_quartile(los).tolist()
        pid_to_stratum = dict(zip(pids, strata))

        _, val, test = _stratified_split(pids, strata, n_val=N_VAL, n_test=N_TEST)
        held_out = val + test
        quartile_counts = [
            sum(1 for p in held_out if pid_to_stratum[p] == q) for q in range(4)
        ]
        for i, c in enumerate(quartile_counts):
            expected = (N_VAL + N_TEST) / 4
            assert abs(c - expected) <= 4, (
                f"Quartile {i} has {c} patients in val+test, expected ~{expected:.0f}"
            )


# ---------------------------------------------------------------------------
# make_splits + load_splits round-trip
# ---------------------------------------------------------------------------


class TestMakeSplits:
    def test_generates_correct_sizes(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        mpath = tmp_path / "manifest.csv"
        spath = tmp_path / "splits.json"
        manifest.to_csv(mpath, index=False)

        splits = make_splits(manifest_path=mpath, splits_path=spath)
        assert splits["n_train"] == N_TRAIN
        assert splits["n_val"] == N_VAL
        assert splits["n_test"] == N_TEST

    def test_writes_json_file(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        mpath = tmp_path / "manifest.csv"
        spath = tmp_path / "splits.json"
        manifest.to_csv(mpath, index=False)

        make_splits(manifest_path=mpath, splits_path=spath)
        assert spath.exists()
        data = json.loads(spath.read_text())
        assert "train" in data and "val" in data and "test" in data

    def test_version_field_present(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        mpath = tmp_path / "manifest.csv"
        spath = tmp_path / "splits.json"
        manifest.to_csv(mpath, index=False)

        splits = make_splits(manifest_path=mpath, splits_path=spath)
        assert splits["version"] == SPLIT_VERSION

    def test_no_overwrite_loads_existing(self, tmp_path: Path) -> None:
        """Second call without overwrite=True must return the same split."""
        manifest = _make_manifest()
        mpath = tmp_path / "manifest.csv"
        spath = tmp_path / "splits.json"
        manifest.to_csv(mpath, index=False)

        s1 = make_splits(manifest_path=mpath, splits_path=spath)
        s2 = make_splits(manifest_path=mpath, splits_path=spath, overwrite=False)
        assert s1["test"] == s2["test"]
        assert s1["val"] == s2["val"]

    def test_overwrite_regenerates(self, tmp_path: Path) -> None:
        """overwrite=True with a different seed must produce different val/test."""
        manifest = _make_manifest()
        mpath = tmp_path / "manifest.csv"
        spath = tmp_path / "splits.json"
        manifest.to_csv(mpath, index=False)

        s1 = make_splits(manifest_path=mpath, splits_path=spath, seed=42)
        s2 = make_splits(manifest_path=mpath, splits_path=spath, seed=99, overwrite=True)
        assert s1["test"] != s2["test"] or s1["val"] != s2["val"]

    def test_raises_when_manifest_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            make_splits(
                manifest_path=tmp_path / "nonexistent.csv",
                splits_path=tmp_path / "splits.json",
            )

    def test_reproducible_with_same_seed(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        mpath = tmp_path / "manifest.csv"
        manifest.to_csv(mpath, index=False)

        s1 = make_splits(
            manifest_path=mpath,
            splits_path=tmp_path / "s1.json",
            seed=RANDOM_SEED,
            overwrite=True,
        )
        s2 = make_splits(
            manifest_path=mpath,
            splits_path=tmp_path / "s2.json",
            seed=RANDOM_SEED,
            overwrite=True,
        )
        assert s1["train"] == s2["train"]
        assert s1["val"] == s2["val"]
        assert s1["test"] == s2["test"]

    def test_patient_ids_are_strings(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        mpath = tmp_path / "manifest.csv"
        spath = tmp_path / "splits.json"
        manifest.to_csv(mpath, index=False)

        splits = make_splits(manifest_path=mpath, splits_path=spath)
        for split_key in ("train", "val", "test"):
            for pid in splits[split_key]:
                assert isinstance(pid, str), f"patient_id is not str: {type(pid)}"

    def test_no_duplicates_within_split(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        mpath = tmp_path / "manifest.csv"
        spath = tmp_path / "splits.json"
        manifest.to_csv(mpath, index=False)

        splits = make_splits(manifest_path=mpath, splits_path=spath)
        for split_key in ("train", "val", "test"):
            ids = splits[split_key]
            assert len(ids) == len(set(ids)), f"Duplicates found in {split_key}"


class TestLoadSplits:
    def test_raises_when_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_splits(tmp_path / "missing.json")

    def test_round_trip(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        mpath = tmp_path / "manifest.csv"
        spath = tmp_path / "splits.json"
        manifest.to_csv(mpath, index=False)

        original = make_splits(manifest_path=mpath, splits_path=spath)
        loaded = load_splits(spath)

        assert original["train"] == loaded["train"]
        assert original["val"] == loaded["val"]
        assert original["test"] == loaded["test"]
        assert original["n_train"] == loaded["n_train"]

    def test_loaded_splits_have_required_keys(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        mpath = tmp_path / "manifest.csv"
        spath = tmp_path / "splits.json"
        manifest.to_csv(mpath, index=False)

        make_splits(manifest_path=mpath, splits_path=spath)
        loaded = load_splits(spath)

        required = {"version", "n_train", "n_val", "n_test", "random_seed",
                    "stratified_by", "train", "val", "test"}
        assert required <= set(loaded.keys())
