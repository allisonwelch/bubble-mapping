"""Turn returned (labeler-edited) grouper packs into correction records +
agreement stats for FUTURE tuning of the learned pairwise 'same-seep?' grouper.

INPUT: packs deployed by deploy_grouper.py >= the snapshot update, i.e.
carrying
  * seep_group_id       -- the labeler's FINAL grouping (edited in QGIS),
  * seep_group_id_pred  -- the FROZEN model proposal at deploy time,
  * group_source        -- 'model' (model-proposed) | 'fixed' (human work the
                           model was told to preserve),
plus the usual labeler / is_calibration / centroid+shape feature columns.

For each within-image candidate pair (<= CAND_RADIUS) among the MODEL-proposed
bubbles it records whether the labeler ACCEPTED / SPLIT / MERGED the model call:
    model_same = seep_group_id_pred[i] == seep_group_id_pred[j]
    human_same = seep_group_id[i]      == seep_group_id[j]
    accept = model_same == human_same; split = model 1->human 0;
    merge  = model 0->human 1.
Only group_source=='model' pairs are scored -- scoring the model on the human
groupings it was told to preserve, or on its own training labels, is circular.

OUTPUTS (to --out-dir):
  corrections_pairs.csv      one row per candidate pair = the edit record
                             (image, seep_id_i<seep_id_j, dist, model_same,
                              human_same, edit_type, is_calibration, labeler).
  corrections_by_labeler.csv accept/split/merge counts + accept_rate +
                             model pairwise error rate, per labeler.
  training_pairs.csv         FEATURES + y=human_same + weight (corrected pairs
                             upweighted) + dedup key (image,seep_id_i,seep_id_j),
                             ready to APPEND to the grouper corpus. Dedup against
                             the original Katey labels the model already trained
                             on (key collision) so they are not double-counted.
  stdout                     a co-membership Fleiss kappa on the CALIBRATION
                             pairs shared across labelers (--model-as-rater adds
                             the model as an extra rater) -> the "modified Fleiss"
                             for grouping agreement.

NOTE: the three EXISTING quarter packs were grouped BEFORE the snapshot column
was added, so they have no seep_group_id_pred -- this tool is for packs deployed
with the updated deploy_grouper.py. It warns and skips a pack that lacks
the snapshot.

ID-NAMESPACE: groups are only unique within an image; every key here is the
(image, seep_id) / (image, seep_group_id) tuple. Candidate pairs are built
within each image only, so cross-image id reuse can never create a spurious pair.

Usage:
  python -m tools.grouping.grouper_corrections PACK1_grouped.gpkg [PACK2 ...] \
         [--out-dir DIR] [--corrected-weight 3.0] [--model-as-rater]
"""
import os
import sys
import sqlite3
import argparse
import numpy as np
import pandas as pd

from tools.grouping.train_grouper import CAND_RADIUS, FEATURES
from tools.grouping.deploy_grouper import _pair_features

SNAP_COL = "seep_group_id_pred"
READ_COLS = ("fid, image, seep_id, "
             "COALESCE(seep_group_id, seep_id) AS seep_group_id, "
             f"{SNAP_COL}, group_source, labeler, "
             "COALESCE(is_calibration,0) AS is_calibration, "
             "centroid_x_m, centroid_y_m, area_m2, circularity, eccentricity, "
             "solidity, mean_R, mean_G, mean_B")


def _has_snapshot(path):
    con = sqlite3.connect(path)
    cols = {r[1] for r in con.execute('PRAGMA table_info("labels")')}
    con.close()
    return SNAP_COL in cols and "group_source" in cols


def pack_pairs(path, corrected_weight):
    """Build the per-pair correction + training rows for one returned pack."""
    con = sqlite3.connect(path)
    df = pd.read_sql_query(f"SELECT {READ_COLS} FROM labels", con)
    con.close()
    lbl = (df["labeler"].dropna().astype(str).str.strip())
    labeler = lbl.mode().iat[0] if len(lbl) and lbl.ne("").any() \
        else os.path.basename(path)

    crec, trec = [], []
    for im, sub in df.groupby("image"):
        fx = sub["centroid_x_m"].to_numpy(float)
        fy = sub["centroid_y_m"].to_numpy(float)
        fa = sub["area_m2"].to_numpy(float)
        elig = sub[sub["group_source"].astype(str) == "model"]
        pp, feat = _pair_features(elig, fx, fy, fa)
        if len(pp) == 0:
            continue
        pred = elig[SNAP_COL].to_numpy()              # frozen model proposal
        final = elig["seep_group_id"].to_numpy()      # labeler-final
        sid = elig["seep_id"].to_numpy(np.int64)
        cal = elig["is_calibration"].fillna(0).astype(int).to_numpy()
        for k, (p, q) in enumerate(pp):
            ms = bool(pred[p] == pred[q])
            hs = bool(final[p] == final[q])
            edit = "accept" if ms == hs else ("split" if ms else "merge")
            i, j = (int(sid[p]), int(sid[q])) if sid[p] < sid[q] \
                else (int(sid[q]), int(sid[p]))
            crec.append({
                "image": im, "seep_id_i": i, "seep_id_j": j,
                "dist": float(feat["dist"].iat[k]),
                "model_same": int(ms), "human_same": int(hs),
                "edit_type": edit,
                "is_calibration": int(cal[p] and cal[q]),
                "labeler": labeler,
            })
            row = {f: float(feat[f].iat[k]) for f in FEATURES}
            row.update({"image": im, "seep_id_i": i, "seep_id_j": j,
                        "y": int(hs), "labeler": labeler,
                        "weight": corrected_weight if edit != "accept" else 1.0})
            trec.append(row)
    return pd.DataFrame(crec), pd.DataFrame(trec)


def fleiss_kappa(counts):
    """Fleiss' kappa for an (n_items x n_categories) count matrix with a
    uniform number of raters per item."""
    counts = np.asarray(counts, float)
    n_items, _ = counts.shape
    n_raters = counts.sum(1)
    if not np.allclose(n_raters, n_raters[0]) or n_raters[0] < 2:
        return float("nan"), int(n_items), int(n_raters[0]) if n_items else 0
    n = n_raters[0]
    p_j = counts.sum(0) / (n_items * n)
    P_i = (np.square(counts).sum(1) - n) / (n * (n - 1))
    Pbar, Pe = P_i.mean(), np.square(p_j).sum()
    kappa = (Pbar - Pe) / (1 - Pe) if Pe < 1 else float("nan")
    return float(kappa), int(n_items), int(n)


def grouping_fleiss(corr, model_as_rater):
    """Co-membership Fleiss on calibration pairs rated by >=2 labelers.
    Each pair is an item; categories = {same, different}; raters = labelers
    (optionally + the model). Restricts to pairs every rater scored."""
    cal = corr[corr["is_calibration"] == 1]
    if cal.empty:
        return None
    wide = cal.pivot_table(index=["image", "seep_id_i", "seep_id_j"],
                           columns="labeler", values="human_same",
                           aggfunc="first")
    if model_as_rater:
        ms = (cal.groupby(["image", "seep_id_i", "seep_id_j"])["model_same"]
                 .first())
        wide["__model__"] = ms.reindex(wide.index).values
    full = wide.dropna(axis=0, how="any")
    if len(full) == 0 or full.shape[1] < 2:
        return None
    same = full.sum(1).to_numpy()
    diff = full.shape[1] - same
    kappa, n_items, n_rat = fleiss_kappa(np.column_stack([same, diff]))
    return {"kappa": kappa, "n_pairs": n_items, "n_raters": n_rat,
            "raters": list(full.columns)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packs", nargs="+")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--corrected-weight", type=float, default=3.0,
                    help="sample weight for corrected (non-accept) pairs in "
                         "training_pairs.csv (accepted pairs weight 1.0)")
    ap.add_argument("--model-as-rater", action="store_true",
                    help="include the model as an extra rater in the Fleiss")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_c, all_t = [], []
    for p in args.packs:
        if not _has_snapshot(p):
            print(f"[skip] {p}: no '{SNAP_COL}'/'group_source' (deployed before "
                  "the snapshot update); cannot diff model vs human.")
            continue
        c, t = pack_pairs(p, args.corrected_weight)
        print(f"[pack] {os.path.basename(p)}: {len(c)} model pairs "
              f"({int((c['edit_type']!='accept').sum())} corrected)")
        all_c.append(c)
        all_t.append(t)
    if not all_c:
        raise SystemExit("no packs with the snapshot column; nothing to do.")

    corr = pd.concat(all_c, ignore_index=True)
    train = pd.concat(all_t, ignore_index=True)
    corr.to_csv(os.path.join(args.out_dir, "corrections_pairs.csv"), index=False)
    train.to_csv(os.path.join(args.out_dir, "training_pairs.csv"), index=False)

    by = (corr.assign(n=1)
              .groupby("labeler")
              .apply(lambda g: pd.Series({
                  "n_pairs": len(g),
                  "accept": int((g.edit_type == "accept").sum()),
                  "split": int((g.edit_type == "split").sum()),
                  "merge": int((g.edit_type == "merge").sum()),
                  "accept_rate": float((g.edit_type == "accept").mean()),
                  "model_pair_err": float((g.model_same != g.human_same).mean()),
              }), include_groups=False)
              .reset_index())
    by.to_csv(os.path.join(args.out_dir, "corrections_by_labeler.csv"),
              index=False)
    print("\n[by labeler]\n", by.to_string(index=False))

    fk = grouping_fleiss(corr, args.model_as_rater)
    if fk is None:
        print("\n[fleiss] no calibration pairs shared across >=2 raters "
              "(need is_calibration pairs in >=2 returned packs).")
    else:
        print(f"\n[fleiss] co-membership kappa = {fk['kappa']:.3f} on "
              f"{fk['n_pairs']} calibration pairs, {fk['n_raters']} raters "
              f"{fk['raters']}")
    print(f"\n[out] {args.out_dir}: corrections_pairs.csv, "
          "corrections_by_labeler.csv, training_pairs.csv")


if __name__ == "__main__":
    main()