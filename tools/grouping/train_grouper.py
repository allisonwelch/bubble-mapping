"""Learned pairwise 'same-seep?' grouper, trained on ALL THREE final labeler
packs.

Grouping is framed as binary classification over pairs of labeled bubbles:

  * one training row = one pair (i, j) of labeled bubbles from THE SAME LABELER
    AND THE SAME IMAGE, within a candidate radius (historical prior, 0.5 m);
  * target y = 1 iff i and j share the same (labeler, image, seep_group_id) --
    i.e. that labeler grouped them into one seep;
  * features describe the relationship between i and j, including LOCAL DENSITY /
    anchor context computed against that image's FULL bubble field.

Inference = predict P(same) per candidate pair (out-of-fold), keep edges above a
threshold, take connected components -> seeps. Scored on per-class seep COUNT
error (the count-based-flux metric).

Two models are fit and compared: an interpretable ``DecisionTreeClassifier`` and
a ``RandomForestClassifier``.

ID-NAMESPACE SAFETY: groups are keyed on the (labeler, image, seep_group_id)
TRIPLE and candidate pairs are generated within one labeler's one image only, so
neither the cross-image id reuse (4.tif and 38.tif both starting seep_group_id
at 1, 2, ...) nor the cross-labeler one can merge bubbles that do not belong
together.

DATA SOURCES (2026-09-11: everything now comes from final_labeler_packs/)
  * gt_seeps_label_quarters_{prajna,allison,katey}_grouped.gpkg -- 14 units over
    8 chips. These are the same packs the classifier trains on, so the two
    stages can no longer disagree about what the labels are.
  * gt_seeps_label_all_chips.gpkg -- the full 48-chip bubble field, supplying
    the per-image DENSITY field (x, y, area only; its bubble_id space is not
    used). Previously only 39.tif got a full-chip density field and every other
    image got a quarter-truncated one, which made dens_* and anchor_ratio mean
    different things per image. Deployment measures density against everything
    detected in the lake, so the full field is also the honest match.

DUPLICATE LABELINGS
The three labelers overlap on the shared calibration units (21-NE, 38-SW, 4-SE)
and the 39.tif quarters overlap too, so the SAME physical pair of bubbles can be
judged by up to three people. ``(image, bubble_id)`` is consistent across every
pack (verified against gt_bubbles.gpkg), so those copies are identifiable: a
pair's physical identity is ``(image, frozenset{bubble_id_i, bubble_id_j})``.

Each pair row is therefore weighted ``1 / (distinct labelers who produced it)``,
so one physical pair contributes one pair's worth of evidence no matter how many
people judged it, while genuine disagreement survives as a soft target rather
than being reconciled away. Without this, the calibration units -- sampled to
measure kappa, not because they matter more -- would carry up to 3x the
influence of everything else.

For the same reason the 5-fold AUC is GROUPED on that physical-pair id: plain
KFold would put one labeler's copy of a pair in train and another's in test.
Leave-one-image-out is unaffected (copies share an image, so they share a fold)
and is the number to quote -- it is the cross-chip regime deployment runs in.

ROW FILTERING (target/training)
  * processed only: ``class`` non-empty (ungrouped rows have blank class).
  * is_overgrouped == 1: EXCLUDED (a single polygon spanning >1 seep cannot be
    split here; flagged for relabel).
  * is_context == 1: kept in the per-image density field and as candidate-pair
    NEIGHBOURS, but truncated context groups are NOT scored as targets (their
    members live partly outside the quarter). Pairs with a context endpoint are
    dropped from train/eval.
  * is_pregrouped == 1: a polygon that already encloses a whole seep. It stays
    an ELIGIBLE PAIRING ANCHOR (truth shows envelopes anchoring satellites, and
    an envelope can only ever gain neighbours) unless PREGROUP_ANCHORS=0.
"""
import argparse
import datetime as _dt
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from shapely.ops import unary_union
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from collections import Counter
import warnings
from sklearn.exceptions import UndefinedMetricWarning


from tools.paths import CANONICAL_PRED_RELDIR


LAB = os.path.join(CANONICAL_PRED_RELDIR, "labeling") + os.sep
FINAL_PACKS = LAB + "final_labeler_packs" + os.sep
# The same three packs tools.classify.fit_classifier trains on. Keep the two in
# step: a grouping the classifier never saw produces hulls it was not fit for.
LABELERS = ["prajna", "allison", "katey"]
# Full 48-chip bubble field. Density only -- x, y and area are read, bubble_id
# is not, so its independent id space cannot collide with the packs'.
FULL_FIELD = LAB + "gt_seeps_label_all_chips.gpkg"


def _pack_path(who):
    return FINAL_PACKS + f"gt_seeps_label_quarters_{who}_grouped.gpkg"


def _require(path, what):
    if not os.path.exists(path):
        raise SystemExit(f"[fail] missing {what}: {path}")
    return path

# historical_size_priors.json -> "grouping" block.
CAND_RADIUS = 0.5        # candidate_radius_m (p95 major axis): max plausible gap
AGGLOM_CAP_M = 1.0       # agglomeration_major_cap_m (p99): wider clusters bridge >1 seep

# THE seed for every model fit here and in deploy_grouper.train_model. Both
# forests and the tree are deterministic given it, so the same packs plus the
# same seed reproduce the same model bit for bit; `build_artifacts` records it
# in artifacts.json alongside the sklearn version and the pack hashes.
DEFAULT_SEED = 42

# Metrics land beside the labels rather than beside the checkpoint, mirroring
# how the classifier workbook is filed. Note this directory is inside the tree
# the labelers sync -- it is written to, never read by the pipeline.
METRICS_OUT_DIR = os.path.join(CANONICAL_PRED_RELDIR, "labeling",
                               "train_grouper", "metrics")

from tools.flux.rates import FLUX_RATE_ANNUAL as FLUX_RATE

FEATURES = ["dist", "size_ratio", "max_area", "min_area",
            "bright_diff", "dens_025", "dens_050",
            "circ_mean", "ecc_mean", "sol_mean", "circ_diff",
            "loc_max_area", "anchor_ratio"]


def _norm_class(s):
    return s.fillna("").astype(str).str.strip().str.upper()


KEEP_COLS = ["labeler", "image", "bubble_id", "class", "seep_group_id",
             "is_pregrouped", "is_overgrouped", "is_context",
             "centroid_x_m", "centroid_y_m", "area_m2",
             "circularity", "eccentricity", "solidity",
             "mean_R", "mean_G", "mean_B", "geometry"]


def load_labeled():
    """Return (L, fields): the pooled labeled-bubble frame from all three final
    packs, and image -> full per-image density field (x, y, area arrays).

    `L` carries a `labeler` column, because a labeler's grouping is only
    meaningful against their own work -- two people's `seep_group_id` values on
    the same chip are unrelated namespaces.
    """
    frames = []
    for who in LABELERS:
        fp = _require(_pack_path(who), f"final labeler pack for {who!r}")
        g = gpd.read_file(fp, layer="labels")
        g["class"] = _norm_class(g["class"])
        g["labeler"] = who
        for col in ("is_context", "is_pregrouped", "is_overgrouped"):
            g[col] = g[col].fillna(0).astype(int)
        frames.append(g[KEEP_COLS])
    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                                geometry="geometry", crs=frames[0].crs)

    # processed (grouped) rows only
    proc = combined[combined["class"] != ""].copy()
    proc = proc[proc["is_overgrouped"] != 1].copy()      # cannot split -> drop

    # ----- per-image FULL density fields (every bubble on the chip, labeled or
    # not, from the 48-chip field). Uniform across images by construction, and
    # it matches what deploy measures density against.
    ff = gpd.read_file(_require(FULL_FIELD, "full bubble field"), layer="labels")
    fields = {}
    for im in proc["image"].unique():
        sub = ff[ff["image"] == im]
        if sub.empty:
            raise SystemExit(
                f"[fail] {im} has labels but no rows in "
                f"{os.path.basename(FULL_FIELD)}; the density field would be "
                f"empty and every dens_* feature would read 0.")
        fields[im] = (sub["centroid_x_m"].to_numpy(float),
                      sub["centroid_y_m"].to_numpy(float),
                      sub["area_m2"].to_numpy(float))
    return proc.reset_index(drop=True), fields


def build_pairs(L, fields, pregrouped_anchors=True):
    """Within-image candidate pairs among labeled bubbles + their features.

    Returns a dict of arrays. Pair endpoints exclude only CONTEXT bubbles
    (truncated neighbour-quarter groups, not scored). is_pregrouped envelopes
    ARE eligible endpoints: in truth they anchor nearby satellite bubbles (16/83
    chip-39 multi-bubble groups mix an envelope with satellites, incl. the
    13-member one), so they must be able to merge. A single envelope polygon can
    only ever GAIN neighbours, never split, so this is one-directional and safe.
    DENSITY is measured against the full per-image field.

    Pairs are generated within one labeler's one image, so a pair is always two
    bubbles the same person was looking at together. `feat` carries two extra
    columns beyond FEATURES: `pair_uid`, the physical identity of the pair
    across labelers, and `w`, the 1/(distinct labelers) sample weight built from
    it. See the module docstring for why."""
    L = L.copy()
    L["bright"] = (L["mean_R"] + L["mean_G"] + L["mean_B"]) / 3.0
    for col in ("circularity", "eccentricity", "solidity"):
        L[col] = L[col].fillna(L[col].median())
    # global key for human grouping. The labeler is part of it: two people's
    # seep_group_id values on one chip are independent namespaces, so without it
    # their groups would silently merge.
    L["gkey"] = (L["labeler"].astype(str) + "::" + L["image"].astype(str)
                 + "::" + L["seep_group_id"].astype(str))
    L["g_int"] = pd.factorize(L["gkey"])[0]

    rec = {k: [] for k in ("i", "j", "y", "img", "lab", "uid")}
    feat_rows = []
    for (lab, im), sub in L.groupby(["labeler", "image"]):
        idx = sub.index.to_numpy()
        # eligible-to-pair endpoints: owned (not context). Pregrouped envelopes
        # are KEPT as eligible anchors (they group with satellites in truth)
        # unless the ablation toggle excludes them.
        mask = sub["is_context"] != 1
        if not pregrouped_anchors:
            mask &= sub["is_pregrouped"] != 1
        elig = sub[mask]
        ei = elig.index.to_numpy()
        if len(ei) < 2:
            continue
        xy = elig[["centroid_x_m", "centroid_y_m"]].to_numpy(float)
        tree = cKDTree(xy)
        pp = tree.query_pairs(r=CAND_RADIUS, output_type="ndarray")
        if len(pp) == 0:
            continue
        fx, fy, fa = fields[im]
        ftree = cKDTree(np.column_stack([fx, fy]))
        a = elig["area_m2"].to_numpy(float)
        b = elig["bright"].to_numpy(float)
        c = elig["circularity"].to_numpy(float)
        e = elig["eccentricity"].to_numpy(float)
        s = elig["solidity"].to_numpy(float)
        g = elig["g_int"].to_numpy(np.int64)
        bid = elig["bubble_id"].to_numpy(np.int64)
        glob = ei
        for p, q in pp:
            mid = (xy[p] + xy[q]) / 2.0
            d025 = max(0, len(ftree.query_ball_point(mid, 0.25)) - 2)
            d050 = max(0, len(ftree.query_ball_point(mid, 0.50)) - 2)
            nbr = ftree.query_ball_point(mid, 0.50)
            loc_max = float(fa[nbr].max()) if len(nbr) else 0.0
            pair_max = max(a[p], a[q])
            feat_rows.append({
                "dist": float(np.hypot(xy[p, 0] - xy[q, 0], xy[p, 1] - xy[q, 1])),
                "size_ratio": min(a[p], a[q]) / max(a[p], a[q]),
                "max_area": pair_max, "min_area": min(a[p], a[q]),
                "bright_diff": abs(b[p] - b[q]),
                "dens_025": d025, "dens_050": d050,
                "circ_mean": (c[p] + c[q]) / 2.0,
                "ecc_mean": (e[p] + e[q]) / 2.0,
                "sol_mean": (s[p] + s[q]) / 2.0,
                "circ_diff": abs(c[p] - c[q]),
                "loc_max_area": loc_max,
                "anchor_ratio": loc_max / max(pair_max, 1e-9),
            })
            rec["i"].append(int(glob[p]))
            rec["j"].append(int(glob[q]))
            rec["y"].append(int(g[p] == g[q]))
            rec["img"].append(im)
            rec["lab"].append(lab)
            # Physical identity of the pair, labeler-independent. bubble_id is
            # consistent across every pack, so two labelers judging the same two
            # bubbles produce the same uid. Order-free.
            lo, hi = sorted((int(bid[p]), int(bid[q])))
            rec["uid"].append(f"{im}::{lo}::{hi}")
    feat = pd.DataFrame(feat_rows)
    feat["pair_uid"] = rec["uid"]
    feat["labeler"] = rec["lab"]
    # One physical pair contributes one pair's worth of evidence, however many
    # people judged it. Counted over DISTINCT LABELERS, not rows, so it stays
    # correct if one labeler ever contributes a pair twice.
    n_lab = feat.groupby("pair_uid")["labeler"].transform("nunique")
    feat["w"] = 1.0 / n_lab.to_numpy(float)
    return (L, feat, np.array(rec["i"]), np.array(rec["j"]),
            np.array(rec["y"]), np.array(rec["img"]))


# ---------------------------------------------------------------------------
# scoring helpers (per-class seep COUNT error)
# ---------------------------------------------------------------------------
def score(part):
    N_h = part["human"].nunique()
    N_r = part["rule"].nunique()
    frag = int((part.groupby("human")["rule"].nunique() > 1).sum())
    bridge = int((part.groupby("rule")["human"].nunique() > 1).sum())
    dom = part.groupby("rule")["hclass"].agg(lambda s: s.value_counts().index[0])
    rc = dom.value_counts().to_dict()
    return N_h, N_r, frag, bridge, rc


def report(tag, part, N_h, over=None):
    """Print one scoring row and return it, so the console table and the
    metrics CSV can never disagree about what was measured."""
    nh, nr, frag, bridge, rc = score(part)
    extra = f" {over:>6}" if over is not None else ""
    print(f"{tag:>26} {nr:>6} {nr - N_h:>+5} {frag:>5} {bridge:>5} "
          f"{rc.get('A', 0):>4} {rc.get('B', 0):>4} {rc.get('C', 0):>4}{extra}")
    return {"config": tag, "n_seep": int(nr), "n_seep_human": int(N_h),
            "bias": int(nr - N_h),
            "bias_pct": 100.0 * (nr - N_h) / N_h if N_h else float("nan"),
            "fragmented": int(frag), "over_merged": int(bridge),
            "n_A": int(rc.get("A", 0)), "n_B": int(rc.get("B", 0)),
            "n_C": int(rc.get("C", 0)),
            "n_over_cap": None if over is None else int(over)}


def overcap(comp, xy, cap):
    over = 0
    for c in np.unique(comp):
        idx = np.where(comp == c)[0]
        if len(idx) < 2:
            continue
        pts = xy[idx]
        d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
        if d.max() > cap:
            over += 1
    return over


def components_from_edges(n, i, j, keep):
    rows = np.concatenate([i[keep], j[keep]])
    cols = np.concatenate([j[keep], i[keep]])
    g = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    return comp


def components_capped(n, i, j, proba, xy, keep, cap):
    """Diameter-capped agglomeration: merge kept edges in descending P(same),
    rejecting any merge whose cluster centroid-span would exceed `cap`. Mirrors
    deploy_grouper.constrained_cluster but over global node indices."""
    parent = list(range(n))
    members = {k: [k] for k in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    idx = np.where(keep)[0]
    for k in idx[np.argsort(-proba[idx])]:
        a, b = int(i[k]), int(j[k])
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        union = members[ra] + members[rb]
        pts = xy[union]
        if np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)).max() <= cap:
            parent[rb] = ra
            members[ra] = union
            del members[rb]
    roots = np.array([find(k) for k in range(n)])
    _, comp = np.unique(roots, return_inverse=True)
    return comp


def oof_proba(clf, X, y, w, groups, splitter):
    """Out-of-fold P(same), fitting each fold WITH the duplicate weights.

    Hand-rolled rather than `cross_val_predict` because sample_weight only
    reaches `fit` through sklearn's metadata routing, and a silently unweighted
    fit is exactly the failure this weighting exists to prevent.
    """
    out = np.zeros(len(y), float)
    for tr, te in splitter.split(X, y, groups=groups):
        m = clone(clf).fit(X[tr], y[tr], sample_weight=w[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    return out


def mean_auc(clf, X, y, w, groups, splitter):
    """Weighted AUC averaged over non-degenerate folds (a held-out fold with
    all-same or all-different pairs carries no ranking information)."""
    aucs = []
    for tr, te in splitter.split(X, y, groups=groups):
        if len(np.unique(y[te])) < 2:
            continue
        m = clone(clf).fit(X[tr], y[tr], sample_weight=w[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1],
                                  sample_weight=w[te]))
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="random_state for both models (default %(default)s). "
                         "The forests are deterministic given it, so two runs "
                         "on the same packs produce the same model.")
    ap.add_argument("--out-dir", default=METRICS_OUT_DIR,
                    help=f"where the metrics CSVs land (default {METRICS_OUT_DIR})")
    ap.add_argument("--no-csv", action="store_true",
                    help="print only; write nothing")
    args = ap.parse_args()
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    L, fields = load_labeled()
    n_img = L["image"].nunique()
    print(f"loaded {len(L)} labeled bubbles across {n_img} images "
          f"from {len(LABELERS)} packs: {L['image'].value_counts().to_dict()}")
    print(f"  by labeler: {L['labeler'].value_counts().to_dict()}")
    print(f"  pregrouped (eligible anchors): {(L['is_pregrouped']==1).sum()}  "
          f"context (features only): {(L['is_context']==1).sum()}")

    pregrouped_anchors = os.environ.get("PREGROUP_ANCHORS", "1") != "0"
    print(f"  [config] pregrouped_anchors = {pregrouped_anchors}")
    L, feat, ii, jj, y, img = build_pairs(L, fields, pregrouped_anchors)
    print(f"candidate pairs (within {CAND_RADIUS} m, owned endpoints): "
          f"{len(y)} | same-seep={int(y.sum())}  "
          f"different={int((1-y).sum())}")
    if y.sum() < 3:
        print("WARNING: <3 positive pairs -- grouping signal is extremely thin.")

    X = feat[FEATURES].to_numpy(float)
    w = feat["w"].to_numpy(float)
    pair_uid = feat["pair_uid"].to_numpy()
    n_phys_pairs = feat["pair_uid"].nunique()
    n_dup = len(feat) - n_phys_pairs
    print(f"  {n_phys_pairs} physical pairs ({n_dup} are duplicate labelings); "
          f"weights 1/n_labelers sum to {w.sum():.0f}")
    # Where the duplication actually sits -- it is concentrated on the shared
    # calibration units, which is the bias the weights remove.
    share = (pd.DataFrame({"image": img, "w": w}).groupby("image")["w"]
             .agg(["size", "sum"]))
    share["row_pct"] = 100 * share["size"] / share["size"].sum()
    share["eff_pct"] = 100 * share["sum"] / share["sum"].sum()
    print("  pair share by image (raw rows vs weighted):")
    print("    " + share.round(1).to_string().replace("\n", "\n    "))
    n = len(L)

    # ----- the SCORED population: owned, non-overgrouped seeps (pregrouped are
    #       fixed singletons & counted; context excluded). -----
    scored_mask = (L["is_context"] != 1).to_numpy()
    hg = L["g_int"].to_numpy(np.int64)
    hcls = L["class"].to_numpy(object)
    sxy = L[["centroid_x_m", "centroid_y_m"]].to_numpy(float)

    def restrict(comp):
        part = pd.DataFrame({"human": hg[scored_mask], "rule": comp[scored_mask],
                             "hclass": hcls[scored_mask]})
        return part

    N_h = int(pd.Series(hg[scored_mask]).nunique())
    hc = (pd.DataFrame({"g": hg[scored_mask], "c": hcls[scored_mask]})
          .drop_duplicates("g")["c"].value_counts().to_dict())
    print(f"\nHUMAN (scored): {N_h} labeled seeps | class {hc}")
    print("  (a seep in a shared calibration unit is counted once PER LABELER. "
          "That is\n   deliberate: the rule-side count below is built over the "
          "same population, so\n   the bias and the flux err% are unaffected -- "
          "but N_h is labeled seeps, not\n   physical seeps.)")

    models = {
        "DecisionTree(d4)": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", min_samples_leaf=5,
            random_state=args.seed),
        "RandomForest(400)": RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=args.seed,
            n_jobs=-1),
    }

    # Two CVs, both grouped. The 5-fold is grouped on the PHYSICAL PAIR so a
    # pair judged by two labelers cannot sit in train and test at once; LOIO is
    # grouped on the image and is the cross-chip regime deployment runs in.
    # The out-of-fold probabilities that drive the seep-count scoring below come
    # from LOIO, not the 5-fold: the 5-fold shares chips between train and test,
    # so it reads optimistic (CLAUDE.md).
    pair_groups = pd.factorize(pair_uid)[0]
    img_groups = pd.factorize(img)[0]
    n_img_groups = len(np.unique(img_groups))
    cv5 = GroupKFold(n_splits=5)
    loio = GroupKFold(n_splits=n_img_groups)

    print(f"\n{'model':>20} {'5f AUC':>7} {'LOIO AUC':>9}   "
          f"(5f grouped on physical pair; LOIO on image, {n_img_groups} chips)")
    oof = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        auc_rows = []
        for name, clf in models.items():
            auc5 = mean_auc(clf, X, y, w, pair_groups, cv5)
            aucg = mean_auc(clf, X, y, w, img_groups, loio)
            print(f"{name:>20} {auc5:>7.3f} {aucg:>9.3f}")
            oof[name] = oof_proba(clf, X, y, w, img_groups, loio)
            auc_rows.append({"model": name, "auc_5fold_grouped_pair": auc5,
                             "auc_loio_image": aucg})

    # decision-tree interpretable rules
    dt = clone(models["DecisionTree(d4)"]).fit(X, y, sample_weight=w)
    print("\n--- DecisionTree(d4) learned rules ---")
    print(export_text(dt, feature_names=FEATURES, max_depth=4))
    print("feature importances: " + ", ".join(
        f"{f}={imp:.2f}" for f, imp in
        sorted(zip(FEATURES, dt.feature_importances_), key=lambda t: -t[1])
        if imp > 0))
    rf = clone(models["RandomForest(400)"]).fit(X, y, sample_weight=w)
    print("\nRandomForest importances: " + ", ".join(
        f"{f}={imp:.2f}" for f, imp in
        sorted(zip(FEATURES, rf.feature_importances_), key=lambda t: -t[1])[:6]))

    print(f"\n{'method':>26} {'N_seep':>6} {'bias':>5} {'frag':>5} {'omrg':>5} "
          f"{'A':>4} {'B':>4} {'C':>4} {'>cap':>6}")
    metric_rows = [report("HUMAN grouping", pd.DataFrame(
        {"human": hg[scored_mask], "rule": hg[scored_mask],
         "hclass": hcls[scored_mask]}), N_h,
        overcap(hg[scored_mask], sxy[scored_mask], AGGLOM_CAP_M))]
    metric_rows[0].update({"model": "HUMAN", "threshold": None})

    # diameter cap (historical agglomeration major-axis cap). ENFORCE_CAP=0
    # reverts to plain connected-components for the capped-vs-uncapped A/B.
    enforce_cap = os.environ.get("ENFORCE_CAP", "1") != "0"
    print(f"  [config] enforce_cap = {enforce_cap} (cap={AGGLOM_CAP_M} m)")

    # for each model, keep the threshold whose seep COUNT is closest to human
    best = {}        # name -> (thr, comp)
    for name in models:
        cand = []
        for thr in (0.5, 0.6, 0.7):
            keep = oof[name] >= thr
            if enforce_cap:
                comp = components_capped(n, ii, jj, oof[name], sxy, keep,
                                         AGGLOM_CAP_M)
            else:
                comp = components_from_edges(n, ii, jj, keep)
            nr = pd.Series(comp[scored_mask]).nunique()
            row = report(f"{name} thr={thr:.1f}", restrict(comp), N_h,
                         overcap(comp[scored_mask], sxy[scored_mask],
                                 AGGLOM_CAP_M))
            row.update({"model": name, "threshold": thr})
            metric_rows.append(row)
            cand.append((abs(nr - N_h), thr, comp))
        cand.sort(key=lambda t: t[0])
        best[name] = (cand[0][1], cand[0][2])
        print()

    # ----- flux-weighted re-score (oracle class) -----
    print("=== flux-weighted re-score (oracle human class; rates A={A:.0f} "
          "B={B:.0f} C={C:.0f} mg CH4/day) ===".format(**FLUX_RATE))
    print(f"  {'config':>26} {'A':>4} {'B':>4} {'C':>4} {'flux':>8} {'err%':>7}")

    def class_counts_of(comp):
        df = pd.DataFrame({"rule": comp[scored_mask], "hclass": hcls[scored_mask]})
        dom = df.groupby("rule")["hclass"].agg(lambda s: s.value_counts().index[0])
        c = Counter(dom)
        return {k: int(c.get(k, 0)) for k in ("A", "B", "C")}

    def flux(c):
        return sum(c[k] * FLUX_RATE[k] for k in ("A", "B", "C"))

    truth = {k: int(hc.get(k, 0)) for k in ("A", "B", "C")}
    f_truth = flux(truth)
    rows = [("TRUTH (human grouping)", truth)]
    for name in models:
        thr, comp = best[name]
        rows.append((f"{name} thr={thr:.1f}", class_counts_of(comp)))
    for tag, c in rows:
        f = flux(c)
        err = 100.0 * (f - f_truth) / f_truth if f_truth else float("nan")
        print(f"  {tag:>26} {c['A']:>4} {c['B']:>4} {c['C']:>4} {f:>8.0f} {err:>+6.1f}%")
    print("  (class is the oracle human label here -- this isolates GROUPING")
    print("   error from classifier error.)")
    print(f"\n(target N_seep={N_h}, bias 0, low omrg, A/B/C ~ {hc})")

    # ----- metrics CSVs -------------------------------------------------- #
    # Every row carries its own flux and err%, computed from the SAME A/B/C
    # counts the console table printed, so the file cannot drift from the log.
    for r in metric_rows:
        f = flux({k: r[f"n_{k}"] for k in ("A", "B", "C")})
        r["flux_mg_CH4_per_day"] = f
        r["flux_err_pct"] = (100.0 * (f - f_truth) / f_truth
                             if f_truth else float("nan"))
        r["selected_threshold"] = (
            r.get("threshold") is not None
            and r["model"] in best
            and abs(best[r["model"]][0] - r["threshold"]) < 1e-9)
    metrics = pd.DataFrame(metric_rows)[[
        "model", "threshold", "config", "selected_threshold",
        "n_seep", "n_seep_human", "bias", "bias_pct", "fragmented",
        "over_merged", "n_A", "n_B", "n_C", "n_over_cap",
        "flux_mg_CH4_per_day", "flux_err_pct"]]
    metrics = metrics.merge(pd.DataFrame(auc_rows), on="model", how="left")

    run_info = pd.DataFrame(
        [{"key": "run_utc", "value": _dt.datetime.now(_dt.timezone.utc)
            .strftime("%Y-%m-%d %H:%M:%S UTC")},
         {"key": "seed", "value": args.seed},
         {"key": "packs", "value": ", ".join(
             os.path.basename(_pack_path(p)) for p in LABELERS)},
         {"key": "density_field", "value": os.path.basename(FULL_FIELD)},
         {"key": "features", "value": ", ".join(FEATURES)},
         {"key": "cand_radius_m", "value": CAND_RADIUS},
         {"key": "agglom_cap_m", "value": AGGLOM_CAP_M},
         {"key": "enforce_cap", "value": enforce_cap},
         {"key": "pregrouped_anchors", "value": pregrouped_anchors},
         {"key": "sample_weight", "value": "1/(distinct labelers per pair_uid)"},
         {"key": "n_labeled_bubbles", "value": int(len(L))},
         {"key": "n_images", "value": int(n_img)},
         {"key": "n_pairs", "value": int(len(y))},
         {"key": "n_physical_pairs", "value": int(n_phys_pairs)},
         {"key": "n_duplicate_pairs", "value": int(n_dup)},
         {"key": "n_same_seep_pairs", "value": int(y.sum())},
         {"key": "n_scored_seeps_human", "value": int(N_h)},
         {"key": "human_class_balance", "value": str(hc)},
         {"key": "flux_truth_mg_CH4_per_day", "value": f_truth}])

    imp = pd.DataFrame(
        [{"model": "DecisionTree(d4)", "feature": f, "importance": float(v)}
         for f, v in zip(FEATURES, dt.feature_importances_)]
        + [{"model": "RandomForest(400)", "feature": f, "importance": float(v)}
           for f, v in zip(FEATURES, rf.feature_importances_)]
    ).sort_values(["model", "importance"], ascending=[True, False])

    if not args.no_csv:
        os.makedirs(args.out_dir, exist_ok=True)
        stem = os.path.join(args.out_dir, f"grouper_results_{stamp}")
        for suffix, frame in (("metrics", metrics), ("run_info", run_info),
                              ("feature_importance", imp),
                              ("pair_share_by_image", share.reset_index())):
            fp = f"{stem}_{suffix}.csv"
            frame.to_csv(fp, index=False)
            print(f"[out] {fp}")


if __name__ == "__main__":
    main()
