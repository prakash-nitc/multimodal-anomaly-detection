# -*- coding: utf-8 -*-
"""Export the per-frame data behind the sealed UBnormal result, for Paper 2's figures.

The sealed run (sealed_ubnormal.py, committed result 2764f93) kept only summary
numbers. ROC curves, a per-scene breakdown and a per-anomaly-type analysis need the
per-frame scores and labels, and the server copy of UBnormal is about to be deleted.

This script does NOT re-run the sealed test and writes nothing to the tables folder.
It recomputes the same four conditions with the sealed script's own functions,
refuses to save unless every micro AUROC equals the committed sealed result to
1e-9, and writes everything else needed for later analysis as plain numpy arrays:
frame embeddings, the text embeddings of every phrase, and the labels -- so no
model and no server are needed afterwards.

UBnormal is CC BY-NC-ND: the output holds labels and must stay OUT of every git
repository (copy it to P:/Research/local_data/, which is not a repo).

    python notebooks/export_ubnormal_scores.py --root ~/dazvad/data/ubnormal
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sealed_ubnormal as su                                       # noqa: E402
from sealed_ubnormal import ns, ub, ideas                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/dazvad/data/ubnormal")
    ap.add_argument("--out", default="~/ubnormal_paper2_export.npz")
    a = ap.parse_args()

    sealed = json.load(open(su.OUT))
    from measurement_audit import load_cache
    data = ub.UBnormalDataset(a.root)
    data.check_files()
    F_pe, L_s, names = load_cache(su.EMB % "PE-Core-L-14-336")
    F_vit, _L, names_v = load_cache(su.EMB % "ViT-L-14")
    assert names == names_v
    videos = [n.split("/", 1)[1] for n in names]
    gts = [data.full_gt(v) for v in videos]
    enc_pe, enc_vit = ns.text_encoder("pe-l"), ns.text_encoder("vitl")

    sampled = {"B": su.scores(F_pe, enc_pe, "B"), "C": su.scores(F_pe, enc_pe, "C"),
               "pe_plain": su.scores(F_pe, enc_pe, "plain"),
               "vit_plain": su.scores(F_vit, enc_vit, "plain")}
    full = {k: [ub.hold_scores(s, len(g)) for s, g in zip(v, gts)] for k, v in sampled.items()}

    print("reproducing the committed sealed result")
    ok = True
    for k in full:
        got, want = su.micro_auc(full[k], gts), sealed["primary_micro_auroc"][k]
        good = abs(got - want) < 1e-9
        ok &= good
        print("  %-10s %.10f  sealed %.10f  %s" % (k, got, want, "ok" if good else "MISMATCH"))
    if not ok:
        print("does not reproduce the sealed result -- nothing saved")
        return 1

    # text embeddings of every phrase, so any margin can be recomputed with numpy alone
    sets = su.phrase_sets()
    calm_n, bank = sets["B"]
    base_n, base_a = sets["plain"]
    types = list(ideas.BANK.keys())
    bank_by_type = [list(ideas.BANK[t]) for t in types]
    out = {
        "names": np.array(names, dtype=object),
        "scene": np.array([int(ub.scene_of(v)) for v in videos], dtype=np.int16),
        "n_frames": np.array([len(g) for g in gts], dtype=np.int32),
        "gt_full": np.concatenate(gts).astype(np.int8),
        "labels_sampled": np.concatenate([np.asarray(l) for l in L_s]).astype(np.int8),
        "n_sampled": np.array([len(f) for f in F_pe], dtype=np.int32),
        "F_pe": np.concatenate([np.asarray(f) for f in F_pe]).astype(np.float32),
        "F_vit": np.concatenate([np.asarray(f) for f in F_vit]).astype(np.float32),
        "T_pe_calm_normal": enc_pe(calm_n).astype(np.float32),
        "T_pe_bank": enc_pe(bank).astype(np.float32),
        "T_pe_base_normal": enc_pe(base_n).astype(np.float32),
        "T_pe_base_abnormal": enc_pe(base_a).astype(np.float32),
        "T_vit_base_normal": enc_vit(base_n).astype(np.float32),
        "T_vit_base_abnormal": enc_vit(base_a).astype(np.float32),
        "bank_types": np.array(types, dtype=object),
        "bank_type_of_phrase": np.array([i for i, t in enumerate(types) for _ in ideas.BANK[t]], dtype=np.int16),
        "phrases_calm_normal": np.array(calm_n, dtype=object),
        "phrases_bank": np.array([p for ps in bank_by_type for p in ps], dtype=object),
        "step": ub.STEP, "w": su.W, "lam": su.LAM,
    }
    for k in full:
        out["held_" + k] = np.concatenate(full[k]).astype(np.float32)
        out["sampled_" + k] = np.concatenate(sampled[k]).astype(np.float32)
    assert list(out["phrases_bank"]) == list(bank), "bank order differs from BANK_PHRASES"

    path = os.path.expanduser(a.out)
    np.savez_compressed(path, **out)
    print("saved %s  (%.1f MB)" % (path, os.path.getsize(path) / 1024 ** 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
