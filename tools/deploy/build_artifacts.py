# tools/deploy/build_artifacts.py
"""Freeze the two post-hoc models into versioned artifacts beside the checkpoint.

THE RULE THIS EXISTS TO ENFORCE: the runner deploys, it never trains.

Before this, `postproc.run()` called `train_model()` and `fit_deploy_model()` on
every single run, refitting both RandomForests from the raw labeler geopackages.
Nothing was serialized anywhere. That made deployment depend on the training data
being present on the machine, let any pack edit silently change the deployed
model, let sklearn version drift change the forest even at `random_state=42`, and
made an old flux number impossible to reproduce. It also contradicted CLAUDE.md's
own rule that grouping parameters and class thresholds travel with the model
checkpoint.

Run this once per (checkpoint, label set) and commit the result to wherever the
checkpoint lives:

    grouper_rf.joblib      pairwise P(same-seep) forest      tools.grouping
    classifier_rf.joblib   A/B/C seep classifier             tools.classify
    artifacts.json         sklearn version, seeds, sha256 of every input pack,
                           feature lists, training-set size and class balance

`artifacts.json` is the part that makes a number reproducible: the pack hashes
say exactly which labels produced these forests, and the sklearn version says
whether a later load can be trusted to behave identically. `load_artifacts`
warns on a version mismatch rather than failing, because a mismatch is usually
survivable -- but it must never be silent.

After this the HPC needs ZERO labeler packs: ortho, checkpoint, lake polygon,
and these two joblibs.

    python -m tools.deploy.build_artifacts [--out-dir DIR] [--seed 42]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import warnings

import joblib
import sklearn

from tools.paths import CANONICAL_CHECKPOINT_RELPATH

GROUPER_FILE = "grouper_rf.joblib"
CLASSIFIER_FILE = "classifier_rf.joblib"
MANIFEST_FILE = "artifacts.json"


def default_out_dir() -> str:
    """Beside the checkpoint, because that is what these travel with."""
    return os.path.dirname(CANONICAL_CHECKPOINT_RELPATH)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _hash_inputs(paths) -> dict:
    out = {}
    for p in paths:
        if p and os.path.exists(p):
            out[os.path.basename(p)] = {
                "path": os.path.abspath(p),
                "sha256": sha256(p),
                "bytes": os.path.getsize(p),
            }
        else:
            out[os.path.basename(str(p))] = {"path": str(p), "sha256": None,
                                             "missing": True}
    return out


def build(out_dir: str | None = None, seed: int = 42,
          labeling_dir: str | None = None,
          brightness: str | None = None) -> dict:
    """Fit both models once and write them plus the manifest. Returns the manifest.

    `brightness` selects the classifier's brightness convention and is recorded
    in the manifest, because the runner has to re-apply the same one. See
    tools/classify/brightness.py for why that matters and what the two modes
    measure.
    """
    from tools.classify.fit_classifier import (DEFAULT_BRIGHTNESS, LABELERS,
                                               default_labeling_dir,
                                               fit_deploy_model)
    from tools.grouping import train_grouper as tg
    from tools.grouping.deploy_grouper import train_model

    out_dir = out_dir or default_out_dir()
    labeling_dir = labeling_dir or default_labeling_dir()
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("FITTING THE GROUPER")
    print("=" * 72)
    grouper = train_model(seed=seed)
    grouper_info = getattr(grouper, "fit_info_", {})

    print("\n" + "=" * 72)
    print("FITTING THE CLASSIFIER")
    print("=" * 72)
    classifier, classifier_info = fit_deploy_model(
        labeling_dir, seed=seed, brightness=brightness or DEFAULT_BRIGHTNESS)

    # Every file that fed either fit. A flux number is only reproducible if you
    # can tell which labels produced the models behind it.
    packs = [tg._pack_path(who) for who in tg.LABELERS]
    packs += [os.path.join(labeling_dir,
                           f"gt_seeps_label_quarters_{who}_grouped.gpkg")
              for who in LABELERS]
    packs.append(tg.FULL_FIELD)
    inputs = _hash_inputs(sorted(set(os.path.abspath(p) for p in packs)))

    manifest = {
        "built_utc": _dt.datetime.now(_dt.timezone.utc)
                        .strftime("%Y-%m-%d %H:%M:%S UTC"),
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "seed": seed,
        "checkpoint": os.path.abspath(CANONICAL_CHECKPOINT_RELPATH),
        "grouper": {
            "file": GROUPER_FILE,
            "group_threshold": None,   # postproc.GROUP_THR owns the op point
            **grouper_info,
        },
        "classifier": {"file": CLASSIFIER_FILE, **classifier_info},
        "inputs": inputs,
    }

    gp = os.path.join(out_dir, GROUPER_FILE)
    cp = os.path.join(out_dir, CLASSIFIER_FILE)
    joblib.dump(grouper, gp)
    joblib.dump(classifier, cp)
    manifest["grouper"]["sha256"] = sha256(gp)
    manifest["classifier"]["sha256"] = sha256(cp)
    with open(os.path.join(out_dir, MANIFEST_FILE), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print("\n" + "=" * 72)
    print("WROTE")
    print("=" * 72)
    for p in (gp, cp, os.path.join(out_dir, MANIFEST_FILE)):
        print(f"  {p}  ({os.path.getsize(p):,} bytes)")
    print(f"\nsklearn {sklearn.__version__}, seed {seed}, "
          f"{len(inputs)} input pack(s) hashed")
    print(f"grouper:    {grouper_info.get('n_pairs')} pairs "
          f"({grouper_info.get('n_physical_pairs')} physical) over "
          f"{grouper_info.get('n_images')} images")
    print(f"classifier: {classifier_info.get('n_training_seeps')} seeps "
          f"(effective {classifier_info.get('effective_n')}), "
          f"balance {classifier_info.get('class_balance_weighted')}")
    return manifest


def _check_features(manifest: dict) -> None:
    """Refuse artifacts whose feature list no longer matches the live code.

    The forests were fit on an ORDERED column list. If `FEATURES` in
    `train_grouper` or `fit_classifier` is edited and nobody rebuilds, a stale
    joblib keeps predicting: sklearn only checks the feature COUNT, so adding or
    removing one raises, but REORDERING two -- or renaming a feature to another
    of the same arity -- sails straight through and produces confident nonsense.
    That is precisely the silent deploy-the-wrong-model failure this module
    exists to remove, so it is a hard error, not a warning.
    """
    from tools.classify.fit_classifier import FEATURES as CLASS_FEATURES
    from tools.grouping.train_grouper import FEATURES as PAIR_FEATURES

    for key, live in (("grouper", PAIR_FEATURES),
                      ("classifier", CLASS_FEATURES)):
        built = manifest.get(key, {}).get("features")
        if built is not None and list(built) != list(live):
            raise SystemExit(
                f"[fail] {key} artifact is stale: it was fit on\n"
                f"    {list(built)}\n"
                f"  but the code now asks for\n"
                f"    {list(live)}\n"
                "  Order matters -- sklearn would not catch a reordering. "
                "Rebuild with `python -m tools.deploy.build_artifacts`.")


def load_artifacts(art_dir: str | None = None):
    """(grouper, classifier, manifest) from disk. Raises if they are not there.

    The caller is expected to fall back to refitting only when explicitly asked
    to -- a deploy run that silently retrains is the thing this module removes.
    """
    art_dir = art_dir or default_out_dir()
    gp = os.path.join(art_dir, GROUPER_FILE)
    cp = os.path.join(art_dir, CLASSIFIER_FILE)
    mp = os.path.join(art_dir, MANIFEST_FILE)
    missing = [p for p in (gp, cp) if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "[fail] deploy artifacts missing:\n  "
            + "\n  ".join(missing)
            + "\nBuild them with `python -m tools.deploy.build_artifacts`, or "
              "pass --refit to fit from the labeler packs (which requires the "
              "packs to be present and makes the run unreproducible).")

    manifest = {}
    if os.path.exists(mp):
        with open(mp) as fh:
            manifest = json.load(fh)
        _check_features(manifest)
        built_with = manifest.get("sklearn_version")
        if built_with and built_with != sklearn.__version__:
            # Survivable but never silent: a pickled forest from another
            # sklearn can load and predict differently.
            warnings.warn(
                f"deploy artifacts were built with scikit-learn {built_with}, "
                f"this environment has {sklearn.__version__}. Predictions may "
                f"not match the run these artifacts were validated on.",
                RuntimeWarning, stacklevel=2)
            print(f"[artifacts] WARNING sklearn {built_with} -> "
                  f"{sklearn.__version__}")
    return joblib.load(gp), joblib.load(cp), manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=None,
                    help=f"default: beside the checkpoint ({default_out_dir()})")
    ap.add_argument("--labeling-dir", default=None,
                    help="the three final labeler packs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--brightness", default=None, choices=("abs", "rel"),
                    help="classifier brightness convention; recorded in the "
                         "manifest and re-applied by the runner")
    args = ap.parse_args(argv)
    build(out_dir=args.out_dir, seed=args.seed, labeling_dir=args.labeling_dir,
          brightness=args.brightness)


if __name__ == "__main__":
    main()
