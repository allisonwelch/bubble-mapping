# tools/deploy/chain.py
"""The post-hoc chain, in one place: bubbles -> screened -> grouped -> classified.

Everything between `bubbles.gpkg` and a per-seep class lives here, and there is
exactly one implementation of it. `postproc.run` calls it to produce the deploy
point estimate; `uncertainty_labels` and `uncertainty_proba` call it to produce
the draws behind an error bar. That single-implementation rule is the whole
point of the module.

THE CENTRING CONTRACT. `run_chain(..., rng=None, wobble=())` is deterministic
and reproduces the deploy number exactly. Every perturbation is opt-in, and each
one must be a spread AROUND that number rather than a different estimator:

    * The grouper's P(same) threshold is HELD at the deploy operating point in
      every draw. Firing each candidate edge at its own probability instead is a
      different question, not a perturbation of this one -- merging is monotone
      under union-find, so it merges systematically more and drops the seep
      count by about a third.
    * The classifier's decision rule (`argmax`, or `conservative`) is HELD in
      every draw, for the same reason: sampling the posterior reproduces the
      model's marginal, while argmax deliberately does not.

What moves instead is the MODEL: `resample_trees` draws the forest's own trees
with replacement, so a draw asks "how firm is this forest's answer?" and the
unperturbed draw is the deploy run by construction. This is a bootstrap over
ensemble members therefore could understate uncertainty given different
training data.

"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import shapely
from shapely.ops import unary_union
from tqdm import tqdm

from tools.classify import brightness as _brightness
from tools.classify.fit_classifier import (CLASSES,
                                           FEATURES as CLASS_FEATURES,
                                           decide_with_cost)
from tools.flux import field_reference
from tools.grouping.deploy_grouper import _pair_features, constrained_cluster
from tools.grouping.train_grouper import AGGLOM_CAP_M, FEATURES as PAIR_FEATURES

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None

GROUP_THR = 0.6   # RF P(same) operating point, per deploy_grouper

# --------------------------------------------------------------------------- #
# THE CRACK SCREEN
# --------------------------------------------------------------------------- #
# The diameter cap constrains MERGES -- the span of a cluster the grouper is
# about to form. It says nothing about a single connected component, so one
# ice crack detected as one long blob sails through it untouched. On the
# 2026-09-14 lake run every seep wider than 3 m was a single CC, and most CCs
# wider than the field maximum were long and thin.
#
# TWO gates, and both must fire: a component has to be LONG and THIN to go.
#
# The shape gate measures thinness as
#
#     thinness = span / (2 * radius of the largest inscribed circle)
#
# which is the width of the fattest place in the component, compared against
# its length. For a rectangle or an ellipse this equals major/minor exactly, so
# it is the same quantity the field workbooks bound -- but it is local, so a
# component that is fat ANYWHERE is not thin, whatever the rest of it does.
# That is the property the two earlier gates lacked:
#
#   perimeter^2/(4 pi area)  measures ROUGHNESS, not width. A rosette of
#       touching bubbles scores as high as a crack, so ragged real clusters
#       went in the bin. This is what dropped 142 of 207 components on
#       2026-09-15.
#   minimum-rotated-rectangle aspect  measures the BOUNDING width. It reads a
#       curved crack as fat (missed) and a straight run of bubbles as thin
#       (dropped), because neither shape fills its own rectangle.
#
# Thresholds:
#   span      1.30 m, the largest major axis among the 2429 hand-measured field
#             seeps in tools/flux/field_reference. Not tuned against a flux
#             total. Measured the same way on both sides: the longest side of
#             the minimum rotated rectangle.
#   thinness  8.0. The field's own aspect bound is 4.0, but that is measured on
#             a whole seep ENVELOPE, and a connected component is a piece of
#             one, so 4.0 cannot be carried across unchanged. 8.0 is the
#             detected bubble population's own extreme upper tail: over 20000
#             components sampled from the 2026-09-15 lake run, the 99.9th
#             percentile of thinness is 5.6 and the 99.99th is 7.6. A component
#             at 8.0 is thinner than essentially every bubble the detector
#             finds, and twice as elongated as any seep the field ever
#             measured.
#
# PROVISIONAL: 8.0 comes from that population tail, not from labeled cracks.
# Confirm it against a hand-labeled set of the wide components before the
# screen's drop rate goes in a manuscript.
#
# Components over the span limit that are not thin are kept and flagged
# `oversized_kept`: a long but fat blob is likelier to be a real feature the
# detector merged than a crack.
SCREEN_MAX_SPAN_M = field_reference.MAX_SEEP_MAJOR_AXIS_M
SCREEN_MIN_THINNESS = 8.0

# --------------------------------------------------------------------------- #
# THE DECISION RULE -- posterior -> class label
# --------------------------------------------------------------------------- #
# The forest emits a posterior; turning that into one label is a separate
# choice, and it is the cheapest lever on the reported flux. Made explicit here
# rather than left implicit inside `clf.predict`, so a run records which rule
# produced its number.
#
#   argmax        the highest-posterior class. Bayes-optimal under 0/1 loss --
#                 it treats calling a C an A exactly as badly as calling an A a
#                 C. Flux does not: those cost 16 and 971 mg CH4/day.
#   conservative  minimum expected cost over the ORDERED classes, with
#                 over-calling penalised by `overcall_penalty`. Errs toward the
#                 smaller class.
#
# Both use the same fitted forest, so switching needs no refit and no new
# labels. Measured effect on the LOIO eval is in SECRET_CLAUDE.md section 3.
DECISION_RULES = ("argmax", "conservative")
DEFAULT_DECISION_RULE = "argmax"
DEFAULT_OVERCALL_PENALTY = 1.5

# Which models a draw may perturb. Named rather than boolean so a run records
# what was sampled, and so a one-at-a-time variance decomposition is a flag.
WOBBLE_TERMS = ("grouper", "classifier")


# --------------------------------------------------------------------------- #
# tree resampling -- the only stochastic element in the models
# --------------------------------------------------------------------------- #
def resample_trees(forest, rng):
    """A copy of `forest` whose trees are drawn with replacement.

    A random forest answers by averaging its trees. Draw the same number of
    trees from it WITH REPLACEMENT and average those instead, and you get a
    plausible alternative version of the same fitted model -- some trees counted
    twice, some left out. Repeat it and the spread of answers says how firm the
    forest's answer is.

    No refit, no training data at run time, no new artifact: the frozen deploy
    forest stays frozen and stays the point estimate, and the perturbation
    exists only to produce the error bar.

    Implemented as a shallow copy with `estimators_` swapped, so prediction
    still goes through sklearn's own vectorised accumulation rather than a
    Python loop over trees. `n_jobs` is forced to 1 because these runs are
    parallelised across draws already, and nested thread pools oversubscribe
    the node.

    Terminology for the writeup: this is a bootstrap over ENSEMBLE MEMBERS, not
    over data. Every tree saw the same packs, so it captures the forest's
    internal averaging noise and NOT the variation that would come from having
    labeled different chips. Say so rather than implying otherwise.
    """
    trees = getattr(forest, "estimators_", None)
    if not trees:
        raise TypeError(
            f"{type(forest).__name__} has no fitted `estimators_`, so its "
            "trees cannot be resampled. Tree resampling needs a fitted "
            "forest-like model.")
    n = len(trees)
    idx = rng.integers(0, n, n)
    sub = copy.copy(forest)
    sub.estimators_ = [trees[i] for i in idx]
    sub.n_estimators = n
    sub.n_jobs = 1
    return sub


def _maybe_wobble(forest, name, wobble, rng):
    """`forest` itself unless `name` is being sampled in this draw."""
    if name not in wobble:
        return forest
    if rng is None:
        raise ValueError(
            f"wobble includes {name!r} but no rng was given; a sampled draw "
            "needs a seeded generator so it can be replayed")
    return resample_trees(forest, rng)


# --------------------------------------------------------------------------- #
# screening
# --------------------------------------------------------------------------- #
def _mrr_axes_m(geom) -> tuple[float, float]:
    """(major, minor) side of the minimum rotated rectangle, in metres.

    The major axis is a shape's true width, not the equivalent-circle diameter,
    which hides exactly the case this screen is for: a 3 m crack with a small
    area reads as a 10 cm circle. Their ratio is elongation, which is what
    separates a crack from a ragged cluster of real bubbles.
    """
    mrr = geom.minimum_rotated_rectangle
    ring = getattr(mrr, "exterior", None)
    if ring is None:                       # degenerate: a point or a line
        return float(mrr.length), 0.0
    xs, ys = ring.coords.xy
    sides = [float(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]))
             for i in range(4)]
    return max(sides), min(sides)


def _max_inscribed_radius_m(geom) -> float:
    """Radius of the largest circle that fits inside `geom`, in metres.

    This is the half-width of the fattest place in the shape, so it is the one
    measurement that says whether a component is circular ANYWHERE. Shapely
    returns the circle as a centre-to-boundary segment, whose length is the
    radius.
    """
    try:
        return float(shapely.maximum_inscribed_circle(geom).length)
    except Exception:            # degenerate geometry: a point or a line
        return 0.0


def screen_bubbles(bubbles, max_span_m=SCREEN_MAX_SPAN_M,
                   min_thinness=SCREEN_MIN_THINNESS, progress=True):
    """Drop crack-like connected components. Returns (kept, dropped).

    A component goes only if it clears both gates -- longer than `max_span_m`,
    and thinner than 1:`min_thinness` at its widest point. Anything that clears
    the span gate alone is KEPT and flagged `oversized_kept`, so a wide real
    feature stays in the flux.

    `dropped` carries `span_m`, `inradius_m`, `thinness`, `aspect`,
    `shape_index` and `screen_reason`, and is written out so the screen can be
    eyeballed in QGIS -- a screen nobody can audit is a screen nobody should
    trust. Only `span_m` and `thinness` drive the decision; `aspect` and
    `shape_index` are carried as diagnostics, because they are what the screen
    used to gate on and a reviewer needs to see both readings on the same rows.

    Deterministic: this is a geometry filter with field-anchored thresholds, not
    a model, so it runs identically in the point estimate and in every draw.
    """
    if max_span_m is None:
        return bubbles, bubbles.iloc[:0].copy()

    geoms = list(tqdm(bubbles.geometry.values, desc="screen",
                      disable=not progress))
    axes = np.array([_mrr_axes_m(g) for g in geoms])
    span, minor = axes[:, 0], axes[:, 1]
    inradius = np.array([_max_inscribed_radius_m(g) for g in geoms])
    with np.errstate(divide="ignore", invalid="ignore"):
        thinness = np.where(inradius > 0, span / (2 * inradius), np.inf)
        aspect = np.where(minor > 0, span / minor, np.inf)
    area = bubbles["area_m2"].to_numpy(dtype=float)
    perim = bubbles["perim_m"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        shape = np.where(area > 0, perim ** 2 / (4 * np.pi * area), np.inf)

    wide = span > max_span_m
    crack = wide & (thinness >= min_thinness)
    flagged = wide & ~crack

    def _annotate(mask, reason):
        out = bubbles[mask].copy()
        out["span_m"] = span[mask]
        out["inradius_m"] = inradius[mask]
        out["thinness"] = thinness[mask]
        out["aspect"] = aspect[mask]
        out["shape_index"] = shape[mask]
        out["screen_reason"] = reason
        return out

    dropped = _annotate(crack, "crack")
    if flagged.any():
        dropped = pd.concat([dropped, _annotate(flagged, "oversized_kept")],
                            ignore_index=True)

    kept = bubbles[~crack].reset_index(drop=True)
    if progress:
        print(f"[screen] {int(crack.sum())} crack-like bubbles removed "
              f"(span > {max_span_m} m AND thinness >= {min_thinness}); "
              f"{int(flagged.sum())} wide but not thin, kept and flagged")
    return kept, dropped


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #
def group_bubbles(clf, bubbles, thr=GROUP_THR, cap=AGGLOM_CAP_M, progress=True):
    """Assign `seep_group_id` within each `image`, anchor-id convention.

    Candidate pairs are generated per image only, so cross-image id reuse can
    never merge bubbles across chips. A whole lake is one image, so the
    partition is the lake and a seep spanning a tile seam groups correctly --
    the tile grid is deliberately not the grouping partition.

    The diameter cap is enforced (`constrained_cluster`): plain connected
    components let chains of short edges bridge into runaway seeps, and flux is
    count-based, so one runaway seep is a real error in the total.

    `thr` is the deploy operating point and is HELD in every Monte Carlo draw.
    It was chosen on labeled data because it reproduces hand-delineated seep
    counts; a sampled edge set has been validated against nothing. Pass a
    different value only to MEASURE sensitivity, and report the spread rather
    than adopting whichever value flatters the total. To perturb the grouper,
    pass a resampled forest as `clf` -- see `resample_trees`.
    """
    # Positional throughout. The previous version wrote via
    # `sgid.iloc[sub.index[m]]`, mixing index LABELS into a positional setter:
    # it happened to be correct only because `run()` resets the index first,
    # and on any other index it either raised or -- worse, when the labels were
    # a permutation that happened to be in range -- silently assigned each
    # group's id to the wrong bubbles.
    sgid = bubbles["bubble_id"].astype(np.int64).to_numpy().copy()
    pos_of = {lbl: i for i, lbl in enumerate(bubbles.index)}
    n_multi = 0
    groups = list(bubbles.groupby("image"))
    for im, sub in tqdm(groups, desc="group", disable=not progress):
        if len(sub) < 2:
            continue
        sub_pos = np.fromiter((pos_of[l] for l in sub.index), int, len(sub))
        fx = sub["centroid_x_m"].to_numpy(float)
        fy = sub["centroid_y_m"].to_numpy(float)
        fa = sub["area_m2"].to_numpy(float)
        pp, feat = _pair_features(sub, fx, fy, fa)
        if len(pp) == 0:
            continue
        proba = clf.predict_proba(feat[PAIR_FEATURES].to_numpy(float))[:, 1]
        keep = proba >= thr
        comp = constrained_cluster(len(sub), pp[keep], proba[keep],
                                   np.column_stack([fx, fy]), cap)
        ids = sub["bubble_id"].to_numpy(np.int64)
        for c in np.unique(comp):
            m = np.where(comp == c)[0]
            sgid[sub_pos[m]] = int(ids[m].max())          # collision-safe anchor
            if len(m) > 1:
                n_multi += 1
    sgid = pd.Series(sgid, index=bubbles.index, name="seep_group_id")
    if progress:
        print(f"[group] {len(bubbles)} bubbles -> "
              f"{bubbles.assign(_g=sgid).groupby(['image', '_g']).ngroups} "
              f"seeps ({n_multi} multi-bubble)")
    return sgid


# --------------------------------------------------------------------------- #
# dissolve
# --------------------------------------------------------------------------- #
def dissolve_bubbles_to_seeps(bubbles, progress=True):
    """One row per (image, seep_group_id): hull geometry + classifier features.

    Mirrors `fit_classifier.dissolve_to_seeps` on every shared term -- hull of
    the unary union, brightness weighted by member area_m2 -- minus the
    labeler-only columns. Read that function before touching this one.
    """
    rows, geoms = [], []
    keys = list(bubbles.groupby(["image", "seep_group_id"], sort=True))
    for (img, gid), sub in tqdm(keys, desc="dissolve", disable=not progress):
        geom = unary_union(sub.geometry.values)
        hull = geom.convex_hull
        w = sub["area_m2"].to_numpy(dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        c = hull.centroid
        rows.append({
            "image": img,
            "seep_group_id": int(gid),
            "n_bubbles": len(sub),
            "area_m2": float(geom.area),
            "hull_area_m2": float(hull.area),
            "perim_m": float(geom.length),
            "centroid_x_m": float(c.x), "centroid_y_m": float(c.y),
            "mean_R": float(np.dot(w, sub["mean_R"].to_numpy(dtype=float))),
            "mean_G": float(np.dot(w, sub["mean_G"].to_numpy(dtype=float))),
            "mean_B": float(np.dot(w, sub["mean_B"].to_numpy(dtype=float))),
        })
        geoms.append(hull)
    seeps = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geoms,
                             crs=bubbles.crs)
    return seeps


# --------------------------------------------------------------------------- #
# classify
# --------------------------------------------------------------------------- #
def classify_seeps(seeps, clf, decision_rule=DEFAULT_DECISION_RULE,
                   overcall_penalty=DEFAULT_OVERCALL_PENALTY):
    """Attach `class` and the per-class posterior columns.

    `decision_rule` selects how the posterior becomes a label; the posterior
    columns (`p_A` / `p_B` / `p_C`) are written either way, so a run can be
    re-decided afterwards without re-running the forest.

    The rule is HELD in every Monte Carlo draw. Drawing a label from the
    posterior instead answers a different question -- it reproduces the model's
    marginal, which argmax deliberately does not -- and an ensemble built that
    way is not centred on the deploy estimate. `uncertainty_proba` samples the
    posterior on purpose, and pairs it with a point estimate computed the same
    way, which is what makes it legitimate there.
    """
    if decision_rule not in DECISION_RULES:
        raise ValueError(f"decision_rule must be one of {DECISION_RULES}, "
                         f"got {decision_rule!r}")
    X = seeps[CLASS_FEATURES].to_numpy(dtype=float)
    seeps = seeps.copy()
    proba = clf.predict_proba(X)
    for i, c in enumerate(clf.classes_):
        seeps[f"p_{c}"] = proba[:, i]

    if decision_rule == "argmax":
        seeps["class"] = clf.predict(X)
    else:
        # Reorder the posterior onto CLASSES, since clf.classes_ is only
        # guaranteed sorted, not equal to CLASSES if a class went unseen.
        cols = {c: i for i, c in enumerate(clf.classes_)}
        p = np.zeros((len(X), len(CLASSES)), dtype=float)
        for j, c in enumerate(CLASSES):
            if c in cols:
                p[:, j] = proba[:, cols[c]]
        seeps["class"] = decide_with_cost(p, list(CLASSES), overcall_penalty)
    return seeps


def posterior_matrix(seeps):
    """The per-seep posterior as an (n, 3) array in CLASSES order.

    A class the forest never saw has no column and reads as zero probability.
    Rows are renormalised, and a row that sums to zero falls back to uniform, so
    downstream sampling never divides by zero.
    """
    p = np.zeros((len(seeps), len(CLASSES)), dtype=float)
    for j, c in enumerate(CLASSES):
        col = f"p_{c}"
        if col in seeps.columns:
            p[:, j] = seeps[col].to_numpy(dtype=float)
    tot = p.sum(axis=1, keepdims=True)
    return np.divide(p, tot, out=np.full_like(p, 1.0 / len(CLASSES)),
                     where=tot > 0)


# --------------------------------------------------------------------------- #
# resolving the operating point
# --------------------------------------------------------------------------- #
def _classifier_brightness(model_prov: dict) -> str:
    """Which brightness convention the loaded classifier was fit under.

    Artifacts built before 2026-09-15 carry no such key. They were fit on
    absolute brightness, so that is the right fallback -- but it is announced,
    because silently guessing wrong would rescale every feature the forest
    splits on and change the class balance, which IS the flux.
    """
    info = (model_prov or {}).get("classifier") or {}
    mode = info.get("brightness")
    if mode is None:
        print("[classify] artifact predates the brightness key; assuming "
              "'abs'. Rebuild with tools.deploy.build_artifacts to record it.")
        return "abs"
    return mode


def _grouper_threshold(model_prov: dict) -> float:
    """The P(same) operating point the loaded grouper was versioned with.

    Artifacts built before 2026-09-15 record `null` here, because the threshold
    used to live only in `GROUP_THR`. Those fall back to the constant, which is
    what they were deployed at -- but it is announced, since the threshold moves
    the seep count and the count is the flux.
    """
    info = (model_prov or {}).get("grouper") or {}
    thr = info.get("group_threshold")
    if thr is None:
        print(f"[group] artifact records no group_threshold; using the module "
              f"default {GROUP_THR}. Rebuild with tools.deploy.build_artifacts "
              f"to version it alongside the model.")
        return GROUP_THR
    return float(thr)


def _default_brightness_cell_m(upstream: dict) -> float:
    """Neighbourhood size for ranking, in metres.

    The detector's own tile size when the run metadata carries it, because that
    is already the scale the imagery was normalized over and the scale the
    classifier's training chips were cut at. Falls back to the module default
    for a bubbles.gpkg that records no grid.
    """
    grid = (upstream or {}).get("grid") or {}
    return float(grid.get("tile_m") or _brightness.DEFAULT_CELL_M)


@dataclass(frozen=True)
class ChainParams:
    """Every setting the chain reads, resolved once and passed down.

    Frozen so a draw cannot mutate the operating point half way through a run,
    and so the same object can be recorded verbatim in the run metadata.
    """
    thr: float
    cap: float = AGGLOM_CAP_M
    brightness_mode: str = "abs"
    brightness_cell_m: float | None = None
    decision_rule: str = DEFAULT_DECISION_RULE
    overcall_penalty: float = DEFAULT_OVERCALL_PENALTY
    max_span_m: float | None = SCREEN_MAX_SPAN_M
    min_thinness: float = SCREEN_MIN_THINNESS
    thr_source: str = "artifact"

    def replace(self, **kw) -> "ChainParams":
        """A copy with some fields changed, for the threshold sweep."""
        return ChainParams(**{**self.__dict__, **kw})

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def resolve_params(model_prov, upstream, *, thr=None, cap=AGGLOM_CAP_M,
                   decision_rule=DEFAULT_DECISION_RULE,
                   overcall_penalty=DEFAULT_OVERCALL_PENALTY,
                   max_span_m=SCREEN_MAX_SPAN_M,
                   min_thinness=SCREEN_MIN_THINNESS,
                   brightness_cell_m=None) -> ChainParams:
    """Build `ChainParams`, deferring to the artifact wherever it records a value.

    `thr=None` means "use the operating point versioned with the model", the
    same way `brightness_cell_m=None` means "use the detector's tile size". An
    explicit value always wins, which is what makes a sensitivity sweep
    possible -- and `thr_source` records which of the two happened, so a run
    never leaves it ambiguous.

    Every caller resolves through here. `deploy.py` used to default its `--thr`
    to the module constant and pass it unconditionally, which made the
    artifact's recorded threshold dead code on the main entry point and live on
    every other one. They agreed only because both read 0.6.
    """
    brightness_mode = _classifier_brightness(model_prov)
    if thr is None:
        thr, thr_source = _grouper_threshold(model_prov), "artifact"
    else:
        thr, thr_source = float(thr), "explicit"
    cell_m = None
    if brightness_mode != "abs":
        cell_m = (float(brightness_cell_m) if brightness_cell_m is not None
                  else _default_brightness_cell_m(upstream))
    return ChainParams(
        thr=thr, cap=float(cap), brightness_mode=brightness_mode,
        brightness_cell_m=cell_m, decision_rule=decision_rule,
        overcall_penalty=float(overcall_penalty),
        max_span_m=(None if not max_span_m else float(max_span_m)),
        min_thinness=float(min_thinness),
        thr_source=thr_source)


# --------------------------------------------------------------------------- #
# the chain
# --------------------------------------------------------------------------- #
@dataclass
class ChainResult:
    """One pass through the chain."""
    bubbles: "pd.DataFrame"      # screened and grouped, `seep_group_id` attached
    screened: "pd.DataFrame"     # what the crack screen dropped or flagged
    seeps: "pd.DataFrame"        # one row per seep, classified, posterior attached
    n_bubbles_in: int            # before screening
    params: ChainParams

    @property
    def n_screened(self) -> int:
        if not len(self.screened):
            return 0
        return int((self.screened["screen_reason"] == "crack").sum())

    def class_counts(self) -> dict:
        """Hard label counts, the population `seeps.gpkg` shows in QGIS."""
        return {c: int((self.seeps["class"] == c).sum()) for c in CLASSES}

    def expected_class_counts(self) -> dict:
        """Posterior-weighted counts: the sum of each seep's probability.

        A seep the forest calls 30% likely to be C contributes 0.3 here and 0
        to `class_counts`. Over many seeps those fractions add up to the
        rare-class seeps a hard label discards, which is why the two differ by
        far more than rounding.
        """
        p = posterior_matrix(self.seeps)
        return {c: float(p[:, j].sum()) for j, c in enumerate(CLASSES)}

    def expected_count_sd(self) -> dict:
        """Standard deviation of each expected count, in closed form.

        Each seep either is or is not class c, independently, with probability
        p_c. A sum of independent yes/no events has variance `sum p(1-p)` --
        the Poisson-binomial distribution -- so no simulation is needed to know
        how much the count would bounce around.

        This assumes seeps are independent of one another. They are not
        entirely: a miscalibrated forest is wrong on many seeps the same way.
        So treat this as a FLOOR on the classifier's contribution, with the
        correlated part living in the held-out-chip confusion matrix, which is
        a bias.
        """
        p = posterior_matrix(self.seeps)
        return {c: float(np.sqrt((p[:, j] * (1.0 - p[:, j])).sum()))
                for j, c in enumerate(CLASSES)}


def run_chain(bubbles, grouper, classifier, params: ChainParams, *,
              progress=True, rng=None, wobble: Sequence[str] = (),
              screened=None) -> ChainResult:
    """screen -> group -> dissolve -> brightness -> classify. One pass.

    Args:
        bubbles: detected bubble polygons + features, any index.
        grouper: the frozen pairwise forest.
        classifier: the frozen A/B/C forest.
        params: resolved operating point, from `resolve_params`.
        progress: progress bars and per-stage prints. Off inside a draw loop.
        rng: numpy Generator, required when `wobble` is non-empty.
        wobble: which models to resample trees for, from `WOBBLE_TERMS`.
        screened: a pre-screened (kept, dropped) pair, to skip re-screening.
            The screen is deterministic, so a draw loop screens once outside it
            and passes the result in.

    Returns:
        ChainResult. With `rng=None, wobble=()` this is the deploy point
        estimate and is bit-identical to what `postproc.run` writes.
    """
    unknown = set(wobble) - set(WOBBLE_TERMS)
    if unknown:
        raise ValueError(f"unknown wobble terms {sorted(unknown)}; "
                         f"expected a subset of {list(WOBBLE_TERMS)}")

    bubbles = bubbles.reset_index(drop=True)
    n_in = len(bubbles)

    # Screen BEFORE grouping: a crack left in is not just a bad seep of its own,
    # it is also a pairing candidate that can pull real bubbles into itself.
    if screened is None:
        bubbles, dropped = screen_bubbles(
            bubbles, max_span_m=params.max_span_m,
            min_thinness=params.min_thinness, progress=progress)
        bubbles = bubbles.reset_index(drop=True)
    else:
        bubbles, dropped = screened
        bubbles = bubbles.reset_index(drop=True)

    grp = _maybe_wobble(grouper, "grouper", wobble, rng)
    cls = _maybe_wobble(classifier, "classifier", wobble, rng)

    bubbles = bubbles.copy()
    bubbles["seep_group_id"] = group_bubbles(
        grp, bubbles, thr=params.thr, cap=params.cap, progress=progress)

    seeps = dissolve_bubbles_to_seeps(bubbles, progress=progress)

    # Re-apply whatever brightness convention the classifier was FIT under.
    # Absolute values would be silently out of scale for a "rel" forest.
    if params.brightness_mode != "abs":
        if progress:
            print(f"[classify] brightness: {params.brightness_mode}, ranked "
                  f"within {params.brightness_cell_m} m cells of the "
                  f"{len(bubbles)} detected bubbles")
        seeps = _brightness.relativize(seeps, bubbles,
                                       mode=params.brightness_mode,
                                       cell_m=params.brightness_cell_m)

    seeps = classify_seeps(seeps, cls, decision_rule=params.decision_rule,
                           overcall_penalty=params.overcall_penalty)
    return ChainResult(bubbles=bubbles, screened=dropped, seeps=seeps,
                       n_bubbles_in=n_in, params=params)
