# File for storing flux rates
"""Per-class seep flux rates and the count-based lake total.

Flux here is COUNT-based:

    total = sum_class ( number of class-X seeps ) * ( per-seep rate for X )

"""
import numpy as np
import pandas as pd

# Flux rates in mg CH4/day based on Walter Anthony et al., 2010 10.4319/lom.2010.8.0592
# Annual Fluxes
FLUX_RATE_ANNUAL = {"A": 16.0, "B": 131.0, "C": 971.0}
FLUX_ANNUAL_STD_ERR = {"A": 10.0, "B": 30.0, "C": 142.0}


#Summer Fluxes
FLUX_RATE_SUMMER = {"A": 28.0, "B": 210.0, "C": 1042.0}
FLUX_SUMMER_STD_ERR = {"A": 19.0, "B": 8.0, "C": 210.0}
FLUX_SUMMER_N = {"A": 3, "B": 2, "C": 6 }

#Winter Fluxes
FLUX_RATE_WINTER = {"A": 8.0, "B": 81.0, "C": 925.0}
FLUX_WINTER_STD_ERR = {"A": 5.0, "B": 29.0, "C": 98.0}
FLUX_WINTER_N = {"A": 8, "B": 4, "C": 9}

CLASSES = ("A", "B", "C")

# Each season is a (rates, standard errors) pair, both in mg CH4/day.
# These are per-DAY rates within each regime, so they are alternative views of
# the same seeps, NOT additive: summer + winter is not a year. Converting a
# regime to a mass needs that regime's length in days, which this module
# deliberately does not assume.
SEASONS = {
    "annual": (FLUX_RATE_ANNUAL, FLUX_ANNUAL_STD_ERR),
    "summer": (FLUX_RATE_SUMMER, FLUX_SUMMER_STD_ERR),
    "winter": (FLUX_RATE_WINTER, FLUX_WINTER_STD_ERR),
}

# Field sample size behind each rate. None for annual (not reported).
#
# The textbook
# small-sample correction, drawing from t(n-1), is unusable here -- at n=2 that
# is df=1, a Cauchy distribution with undefined variance, which would emit
# occasional absurd Monte Carlo draws. `lake_total` uses a moment-matched
# lognormal instead; see its docstring.
SEASON_N = {
    "annual": None,
    "summer": FLUX_SUMMER_N,
    "winter": FLUX_WINTER_N,
}


# --------------------------------------------------------------------------- #
# reporting units
# --------------------------------------------------------------------------- #
# Every flux figure in this repo is computed in mg CH4/day. Two derived units
# are reported beside it, because a bare mg/day total is not comparable to
# anything: per-area makes it comparable to another lake or another flight
# date, and an annual mass makes it comparable to the published literature.
DAYS_PER_YEAR = 365.0
MG_PER_G = 1000.0

# The column suffix every mg CH4/day quantity carries, which is how
# `add_per_area_columns` finds them.
PER_DAY_SUFFIX = "_mg_CH4_per_day"

# How many days of a year each season's rate describes. ONLY THE ANNUAL RATE
# COVERS A WHOLE YEAR, so only it can be integrated to one. Summer and winter
# are per-day rates within a regime whose length the workbooks do not record,
# and this module deliberately does not assume one -- multiplying either by 365
# would report a year made entirely of that regime. Those rows get NaN in the
# per-year column rather than a number that reads like an annual budget.
SEASON_DAYS_PER_YEAR = {"annual": DAYS_PER_YEAR, "summer": None, "winter": None}


def days_per_year(season="annual"):
    """Days of the year this season's per-day rate describes, or None."""
    season_rates(season)  # validates the name
    return SEASON_DAYS_PER_YEAR.get(season)


def per_area_units(value_mg_per_day, surveyed_area_m2, season="annual"):
    """One mg CH4/day figure as (mg/m2/day, g/m2/year).

    The conversion is linear and the surveyed area is a constant, so this
    applies to a percentile, a standard error or a central estimate alike: a
    2.5th percentile of the total is the 2.5th percentile of the density.

    Both are NaN without a surveyed area, and the per-year figure is NaN for
    any season but annual. See `SEASON_DAYS_PER_YEAR`.
    """
    if not surveyed_area_m2:
        return float("nan"), float("nan")
    density = float(value_mg_per_day) / float(surveyed_area_m2)
    days = days_per_year(season)
    if days is None:
        return density, float("nan")
    return density, density * days / MG_PER_G


def add_per_area_columns(df, surveyed_area_m2, season=None,
                         season_col="season", exclude=()):
    """Add per-area twins for every mg CH4/day column in a table.

    Each column named `*_mg_CH4_per_day` gains two siblings immediately to its
    right:

        *_mg_CH4_per_m2_per_day   the same figure over the surveyed area
        *_g_CH4_per_m2_per_year   that density as an annual mass

    Applying this to the whole table rather than to the central estimate alone
    is the point: an interval reported in one unit and a point estimate
    reported in another is how a reader ends up comparing two different
    quantities.

    Args:
        df: any table whose flux columns carry the `_mg_CH4_per_day` suffix.
        surveyed_area_m2: the measured denominator. Falsy leaves `df` unchanged,
            so a missing area never becomes a NaN density that reads as zero.
        season: season for the per-year conversion, used when `df` carries no
            season column.
        season_col: per-row season column, which takes precedence over `season`.
        exclude: columns to leave alone. Pass the PER-SEEP published rates:
            dividing one seep's rate by the whole lake area is meaningless.

    Returns:
        A new DataFrame. The source columns keep their names and values.
    """
    if not surveyed_area_m2:
        return df
    area = float(surveyed_area_m2)

    out = df.copy()
    if season_col in out.columns:
        days = out[season_col].map(SEASON_DAYS_PER_YEAR)
    else:
        days = pd.Series(days_per_year(season or "annual"), index=out.index)
    days = pd.to_numeric(days, errors="coerce")

    targets = [c for c in df.columns
               if c.endswith(PER_DAY_SUFFIX) and c not in set(exclude)]
    for col in targets:
        base = col[:-len(PER_DAY_SUFFIX)]
        density = pd.to_numeric(out[col], errors="coerce") / area
        at = out.columns.get_loc(col) + 1
        out.insert(at, f"{base}_mg_CH4_per_m2_per_day", density)
        out.insert(at + 1, f"{base}_g_CH4_per_m2_per_year",
                   density * days / MG_PER_G)
    return out


def season_rates(season="annual"):
    """(rates, std_errors) for one season. Raises on an unknown name."""
    try:
        return SEASONS[season]
    except KeyError:
        raise KeyError(
            f"unknown season {season!r}; expected one of {sorted(SEASONS)}"
        ) from None


def season_n(season="annual"):
    """Field sample size per class, or None where it was not reported."""
    season_rates(season)  # validates the name
    return SEASON_N.get(season)


def class_sd(season="annual"):
    """Per-seep standard deviation per class, or None without an n.

    SD = SE * sqrt(n). This is the spread among INDIVIDUAL seeps, as opposed
    to the standard error, which is the uncertainty in the class mean. The
    lake total is a class mean times a count, so `lake_total` propagates the
    standard error; the SD only enters the separate seep-to-seep variability
    term reported by `flux_table`.
    """
    _, std_errs = season_rates(season)
    n = season_n(season)
    if n is None:
        return None
    return {c: std_errs[c] * np.sqrt(n[c]) for c in CLASSES}


def _lognormal_draw(rng, mean, sd):
    """One draw from a lognormal with exactly this mean and standard deviation.

    Moment matching: for X ~ LogNormal(mu, s),
        E[X]   = exp(mu + s^2 / 2)
        Var[X] = (exp(s^2) - 1) * exp(2*mu + s^2)
    Solving for (mu, s) given the target mean and sd gives the two lines
    below. Degenerate inputs (sd <= 0, mean <= 0) fall back to the point value.
    """
    if sd <= 0 or mean <= 0:
        return float(mean)
    s2 = np.log1p((sd / mean) ** 2)
    mu = np.log(mean) - s2 / 2.0
    return float(rng.lognormal(mu, np.sqrt(s2)))


def lake_total(class_counts, season="annual", rng=None):
    """Count-based flux total for a set of classified seeps.

    Args:
        class_counts: mapping of class -> seep count, e.g.
            {"A": 100, "B": 20, "C": 4}. Missing classes count as zero, so a
            pandas value_counts() or a collections.Counter both work.
        season: "annual", "summer" or "winter".
        rng: optional numpy Generator. Without it, the total uses the
            published point rates. With it, each class rate is drawn once
            from a LOGNORMAL moment-matched to (rate, std_err), and the total
            is a single Monte Carlo realization.

            One draw per CLASS, not per seep: the standard error belongs to
            the class mean, so the error is shared by every seep of that

            Lognormal rather than normal because emission rates are
            positive-only and right-skewed. A normal draw on annual A
            (16 +/- 10) goes negative about 5.5% of the time, and clipping
            those at zero biases the sampled mean upward by ~0.4%. The
            lognormal has no mass below zero, so nothing needs clipping and
            the mean is preserved exactly.

    Returns:
        (total, rate_sigma), both mg CH4/day. `rate_sigma` is always the
        analytic quadrature value and does not depend on `rng`, so it stays
        comparable between sampled and unsampled calls.
    """
    rates, std_errs = season_rates(season)
    n = {c: int(class_counts.get(c, 0)) for c in CLASSES}

    if rng is None:
        drawn = rates
    else:
        drawn = {c: _lognormal_draw(rng, rates[c], std_errs[c])
                 for c in CLASSES}

    total = float(sum(drawn[c] * n[c] for c in CLASSES))
    rate_sigma = float(np.sqrt(sum((std_errs[c] * n[c]) ** 2 for c in CLASSES)))
    return total, rate_sigma


def flux_table(class_counts, seasons=None):
    """One row per season: counts, total, and the uncertainty terms.

    Two uncertainty columns, deliberately NOT combined into one:

      rate_std_err_*  the standard error on each class mean, propagated as
                      (SE_c * N_c) in quadrature. Fully correlated across
                      seeps, so it does not shrink as more seeps are mapped.
                      This is the published floor.
      seep_var_*      seep-to-seep variability, (SD_c * sqrt(N_c)) in
                      quadrature, with SD_c = SE_c * sqrt(n_field). Averages
                      down as 1/sqrt(N), so it is small for A and B but not
                      for C, where the lake holds few seeps. NaN for annual,
                      which reports no n.

    Neither covers detector, grouper or classifier error; those have no
    closed form and need the Monte Carlo chain.

    This is the runner's end-of-chain summary. Returns a DataFrame; see
    `write_flux_table` to put it on disk.
    """
    rows = []
    for season in (seasons or list(SEASONS)):
        total, sigma = lake_total(class_counts, season=season)
        rates, _ = season_rates(season)
        sds = class_sd(season)
        n = {c: int(class_counts.get(c, 0)) for c in CLASSES}
        row = {"season": season}
        row.update({f"n_{c}": n[c] for c in CLASSES})
        row["n_seeps"] = sum(n.values())
        row["total_mg_CH4_per_day"] = total
        # Per-class contribution: count x rate. These sum to the total, and are
        # the honest way to see that a class holding a few percent of the seeps
        # can hold most of the methane.
        row.update({f"flux_{c}_mg_CH4_per_day": float(rates[c] * n[c])
                    for c in CLASSES})
        row.update({f"rate_{c}_mg_CH4_per_day": float(rates[c])
                    for c in CLASSES})
        row["rate_std_err_mg_CH4_per_day"] = sigma
        row["rate_std_err_pct"] = 100 * sigma / total if total else float("nan")
        if sds is None:
            seep_var = float("nan")
        else:
            seep_var = float(np.sqrt(
                sum((sds[c] ** 2) * n[c] for c in CLASSES)))
        row["seep_var_mg_CH4_per_day"] = seep_var
        row["seep_var_pct"] = 100 * seep_var / total if total else float("nan")
        # C is a few percent of seeps but roughly half the methane, which is
        # why cross-chip C recall outranks the A/B boundary as a flux lever.
        row["pct_flux_from_C"] = (
            100 * rates["C"] * n["C"] / total if total else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def write_flux_table(class_counts, path, seasons=None):
    """Write the per-season summary to .xlsx (needs openpyxl) or .csv."""
    df = flux_table(class_counts, seasons=seasons)
    if str(path).lower().endswith((".xlsx", ".xlsm")):
        df.to_excel(path, index=False, sheet_name="flux")
    else:
        df.to_csv(path, index=False)
    return df
