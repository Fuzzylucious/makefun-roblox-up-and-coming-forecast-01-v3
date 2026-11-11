#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster v3 (make.fun bounty
roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson (snapshots of Roblox's Up-and-Coming chart,
oldest first) and /input/task.json, and writes /output/forecast.json:
per-place probabilities for the top 10 places plus a top-10 guess in
order, scored by log score over the 72 hours after the deadline.

Method: Monte Carlo rank diffusion in latent-appeal space (a = -log rank),
with a regression-calibrated drift, heavy-tailed (Student-t) steps,
phantom entrants, and a 10% blend with the persistence forecast.

What v3 changes over v2 (validated on the same 7 holdout windows --
fit on snapshots 1..cut, scored on the reads in the 72 h after the cut,
cuts 20/24/28/30/34/38/42; v2 stdlib on the same windows: mean -34.80,
median -32.56, cut34 -31.38):

- A NEW DRIFT FEATURE: log-playing LEVEL relative to the chart median
  (lpr = log(playing) - median log(playing) of the last chart).
  Refit OLS on the 3,650 consecutive-pair transitions: adding lpr lifts
  R2 from 0.123 to 0.134 on its own, and with the refit coefficients
  (slope96 0.546, gplay30 0.625, log-rank 0.0116 combined, log-age 0.0007,
  lpr 0.0044, const -0.049) the holdout mean improves to -34.32 and the
  practice window (cut34) to -30.95. A game whose playing count is far
  above its rank peers is a breakout in progress: its appeal drifts up
  even when its rank slope has not moved yet.
- What did NOT survive validation (tested, dropped): a 24 h appeal
  slope (pure noise on sparse reads, -35.47), the playing-acceleration
  pair gp30/gp96 (-34.74), a tenure/creation-age hazard (-34.66),
  rank-band sigma (-35.08), drift decay (-35.01), temperature
  flattening (0.9: -34.58, 0.7: -35.08), and ensembling with a
  slope-drift MC (-35.12). The extended regression's in-sample R2 of
  0.156 does NOT transfer -- the extra features are noise in the MC.
- PERSISTENCE BLEND (weight 0.10): the Monte Carlo forecast is averaged
  with the starter persistence forecast (0.5 on the game at each place,
  0.15 on its neighbours). It wins on all four recent holdout windows
  (recent-4 mean -31.26 vs -31.48 without it) and on the practice
  window (-30.79 vs -30.95); it only loses on the three oldest windows,
  a different regime. The final window's first ~12 hourly reads sit
  close to the last chart, where persistence is strongest.

Standard library only, no network, deterministic (fixed seed).
"""

import json
import math
import os
import random
from datetime import datetime, timedelta

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Tunables (calibrated on the archive by multi-window holdout scoring;
# fit on snapshots 1..cut, score on the reads in the 72 h after the cut,
# cuts 20/24/28/30/34/38/42. Stdlib v3: mean -34.29, median -31.92,
# recent-4 -31.30, practice window (cut34) -30.86; v2 on the same
# windows: mean -34.80, median -32.56, recent-4 -32.05, cut34 -31.38.
# Live practice board: v2 -31.4 at rank 13; rank 10 is -30.9, rank 1 -29.0.)
SEED = 20261113
N_SIMS = 1500           # Monte Carlo paths
DRIFT_WINDOW_HOURS = 96.0
GPLAY_WINDOW_HOURS = 30.0
DRIFT_SHRINK_K = 2.0     # slope *= n / (n + k)
DRIFT_CAP = 0.030        # |appeal slope| per hour, capped
# Regression drift v3: OLS on 3,650 consecutive-pair transitions from
# the archive (R^2 = 0.134 with lpr). E[da/h] = REG_CONST
# + REG_SLOPE * appeal slope (trailing 96 h, capped +-0.03, shrunk
# n/(n+2)) + REG_GPLAY * log-playing slope (trailing 30 h, clipped
# +-0.05) + REG_LOG_RANK * log(rank) [0.0116 = the fit's separate
# appeal (-0.0058) and log-rank (0.0058) terms combined, since appeal
# is -log(rank)] + REG_LOG_AGE * log(chart-age in hours)
# + REG_LPR * (log(playing) - median log(playing) of the last chart).
REG_SCALE = 1.2          # regression drift is shrunk; 1.2 fits best
REG_CONST = -0.049
REG_SLOPE = 0.5462
REG_GPLAY = 0.6252
REG_LOG_RANK = 0.0116
REG_LOG_AGE = 0.0007
REG_LPR = 0.0044
SIGMA = 0.10             # Student-t scale per sqrt-hour
T_DF = 12                # Student-t degrees of freedom
PHANTOM_SIGMA = 0.35     # entrants diffuse faster than incumbents
ENTRANT_DRIFT = 0.012    # entrants keep climbing, decaying with age
ENTRANT_HALFLIFE_H = 24.0
SMOOTH_EPS = 0.02        # blend of raw counts with a flat prior
P_FLOOR = 0.0015         # rows below this are dropped (mass stays unnamed)
P_CAP = 0.65             # no single game-place probability above this
PLACE_SUM_CAP = 0.95     # leave at least 5% of every place unnamed
PERSIST_W = 0.10         # weight of the persistence forecast in the blend


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(input_dir):
    snaps = []
    with open(os.path.join(input_dir, "history.ndjson"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    snaps.sort(key=lambda s: s["at"])
    task = {}
    try:
        with open(os.path.join(input_dir, "task.json"), encoding="utf-8") as f:
            task = json.load(f)
    except OSError:
        pass
    return snaps, task


def _slope(xs, ys):
    """Least-squares slope with n/(n+k) shrinkage; xs in hours."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return 0.0
    s = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return s * n / (n + DRIFT_SHRINK_K)


def estimate_params(snaps):
    """Per-game drift (regression v3, incl. playing-level feature),
    entry rate/ranks, last chart."""
    series = {}
    play_series = {}
    first_seen = {}
    entry_ranks = []
    for idx, snap in enumerate(snaps):
        t = parse_time(snap["at"])
        for row in snap.get("up_and_coming") or []:
            uid = row["universe_id"]
            series.setdefault(uid, []).append((t, row["rank"]))
            if row.get("playing"):
                play_series.setdefault(uid, []).append((t, row["playing"]))
            if uid not in first_seen:
                first_seen[uid] = t
                if idx > 0:
                    entry_ranks.append(row["rank"])
    t_end = parse_time(snaps[-1]["at"])
    last_chart = {r["universe_id"]: (r["rank"], r.get("name") or "")
                  for r in snaps[-1]["up_and_coming"]}
    # Median log-playing of the last chart (the lpr reference level).
    lps = sorted(math.log(max(r.get("playing") or 1, 1))
                 for r in snaps[-1]["up_and_coming"] if r.get("playing"))
    med_log_play = lps[len(lps) // 2] if lps else 0.0

    t_start = t_end - timedelta(hours=DRIFT_WINDOW_HOURS)
    g_start = t_end - timedelta(hours=GPLAY_WINDOW_HOURS)
    last_rank = {uid: rank for uid, (rank, _n) in last_chart.items()}
    drifts = {}
    for uid, obs in series.items():
        recent = [(t, r) for (t, r) in obs if t >= t_start]
        if len(recent) < 3:
            recent = obs[-3:]
        slope = 0.0
        if len(recent) >= 2:
            xs = [(t - recent[0][0]).total_seconds() / 3600.0 for t, _ in recent]
            ys = [-math.log(r) for _, r in recent]
            slope = min(max(_slope(xs, ys), -DRIFT_CAP), DRIFT_CAP)
        # Trailing log-playing slope (growth leads rank by hours).
        gplay = 0.0
        trail = [(t, p) for (t, p) in play_series.get(uid, []) if t >= g_start]
        if len(trail) >= 3:
            xs = [(t - trail[0][0]).total_seconds() / 3600.0 for t, _ in trail]
            ys = [math.log(max(p, 1)) for _, p in trail]
            gplay = min(max(_slope(xs, ys), -0.05), 0.05)
        # Playing LEVEL relative to the chart median (v3 feature):
        # a game with far more players than its rank peers is breaking out.
        last_play = play_series.get(uid, [[None, 0]])[-1][1]
        lpr = (math.log(max(last_play, 1)) - med_log_play) if last_play else 0.0
        age_h = max((t_end - first_seen[uid]).total_seconds() / 3600.0, 1.0)
        rank = last_rank.get(uid, obs[-1][1])
        drift = (REG_CONST + REG_SLOPE * slope + REG_GPLAY * gplay
                 + REG_LOG_RANK * math.log(rank) + REG_LOG_AGE * math.log(age_h)
                 + REG_LPR * lpr)
        drifts[uid] = drift * REG_SCALE
    span_h = 0.0
    if len(snaps) >= 2:
        span_h = (parse_time(snaps[-1]["at"])
                  - parse_time(snaps[0]["at"])).total_seconds() / 3600.0
    entry_rate = min(max(len(entry_ranks) / span_h, 0.05), 1.5) if span_h > 0 else 0.5
    return {
        "drifts": drifts,
        "entry_rate": entry_rate,
        "entry_ranks": sorted(entry_ranks) or [80, 87, 90, 95, 97],
        "last_chart": last_chart,
    }


def simulate(params, places, horizon_hours, n_sims=N_SIMS, seed=SEED):
    """Monte Carlo over the scored window. Returns counts[uid][place]."""
    rng = random.Random(seed)
    last_chart = params["last_chart"]
    drifts = params["drifts"]
    entry_rate = params["entry_rate"]
    entry_ranks = params["entry_ranks"]

    games = []  # [uid, appeal, drift]
    for uid, (rank, _name) in sorted(last_chart.items(), key=lambda kv: kv[1][0]):
        games.append([uid, -math.log(rank), drifts.get(uid, 0.0)])
    n_games = len(games)
    # Phantom entrants: expected count over the window, from the
    # empirical entry rate (about 0.5/hour in the archive).
    n_phantoms = min(max(int(round(entry_rate * horizon_hours)), 25), 70)

    def t_noise():
        # Exact Student-t with T_DF degrees of freedom:
        # Z / sqrt(chi2_df / df), chi2 via 2 * Gamma(df/2, 1).
        z = rng.gauss(0.0, 1.0)
        chi2 = 2.0 * rng.gammavariate(T_DF / 2.0, 1.0)
        return z / math.sqrt(chi2 / T_DF)

    counts = {uid: [0] * (places + 1) for uid, _, _ in games}
    total_slots = n_sims * horizon_hours
    for _ in range(n_sims):
        state = [[g[0], g[1], g[2]] for g in games]
        # Phantoms: [entry_hour, appeal, age_hours]; entry hours uniform
        # over the window, entry appeals from the empirical entry ranks.
        phantoms = []
        for _ in range(n_phantoms):
            r = entry_ranks[rng.randrange(len(entry_ranks))]
            phantoms.append([rng.uniform(0.0, horizon_hours),
                             -math.log(r) + rng.gauss(0, 0.3), 0.0])
        for hour in range(1, horizon_hours + 1):
            scored = []
            for g in state:
                g[1] += g[2] + t_noise() * SIGMA
                scored.append((g[1], g[0]))
            for ph in phantoms:
                if ph[0] <= hour:
                    ph[2] += 1.0
                    decay = 0.5 ** (ph[2] / ENTRANT_HALFLIFE_H)
                    ph[1] += ENTRANT_DRIFT * decay + t_noise() * PHANTOM_SIGMA
                    scored.append((ph[1], -1))
            scored.sort(key=lambda x: -x[0])
            for place in range(1, min(places, len(scored)) + 1):
                uid = scored[place - 1][1]
                if uid >= 0:
                    counts[uid][place] += 1
    return counts, total_slots, n_games


def build_forecast(counts, total_slots, n_games, places, last_chart):
    probs = []
    for place in range(1, places + 1):
        rows = []
        for uid, arr in counts.items():
            p = arr[place] / total_slots if total_slots else 0.0
            # Blend with a flat prior so no plausible game is ever at zero.
            p = (1.0 - SMOOTH_EPS) * p + SMOOTH_EPS * (1.0 / max(n_games, 1))
            if p >= P_FLOOR:
                rows.append((uid, min(p, P_CAP)))
        total = sum(p for _, p in rows)
        if total > PLACE_SUM_CAP:
            scale = PLACE_SUM_CAP / total
            rows = [(uid, p * scale) for uid, p in rows]
        rows.sort(key=lambda x: (-x[1], x[0]))
        for uid, p in rows:
            probs.append({"universe_id": uid, "place": place, "p": round(p, 6)})
    return {"probabilities": probs}


def persistence_forecast(last_chart, places):
    """The starter baseline: the chart as it stood (0.5 on the game at
    each place, 0.15 on its neighbours; 0.3 on #2 at place 1)."""
    order = [uid for uid, (_r, _n) in
             sorted(last_chart.items(), key=lambda kv: kv[1][0])]
    rows = {}
    for place in range(1, places + 1):
        if place - 1 >= len(order):
            break
        rows[(order[place - 1], place)] = 0.5
        if place == 1:
            if len(order) > 1:
                rows[(order[1], 1)] = 0.3
        else:
            if place - 2 >= 0:
                rows[(order[place - 2], place)] = 0.15
            if place < len(order):
                rows[(order[place], place)] = 0.15
    return rows


def blend_persist(forecast, last_chart, places, w=PERSIST_W):
    """Average the Monte Carlo forecast with the persistence forecast
    (weight w). The top 10 is re-derived greedily afterwards."""
    persist = persistence_forecast(last_chart, places)
    mc = {(r["universe_id"], r["place"]): r["p"]
          for r in forecast["probabilities"]}
    out = []
    for (uid, place), p in mc.items():
        out.append({"universe_id": uid, "place": place,
                    "p": round((1.0 - w) * p + w * persist.get((uid, place), 0.0), 6)})
    seen = {(r["universe_id"], r["place"]) for r in out}
    for (uid, place), p in persist.items():
        if (uid, place) not in seen and p > 0:
            out.append({"universe_id": uid, "place": place,
                        "p": round(w * p, 6)})
    out.sort(key=lambda r: (r["place"], -r["p"]))
    # Top 10: greedy by probability, place by place.
    by_place = {}
    for row in out:
        by_place.setdefault(row["place"], []).append(row)
    top10 = []
    used = set()
    for place in range(1, places + 1):
        for row in by_place.get(place, []):
            if row["universe_id"] not in used:
                top10.append(row["universe_id"])
                used.add(row["universe_id"])
                break
    if len(top10) < places:  # fall back to the last chart's order
        for uid, (_rank, _name) in sorted(last_chart.items(), key=lambda kv: kv[1][0]):
            if uid not in used:
                top10.append(uid)
                used.add(uid)
            if len(top10) == places:
                break
    return {"probabilities": out, "top10": top10[:places]}


def main():
    snaps, task = load_history(INPUT_DIR)
    places = int(task.get("places", 10))
    window = task.get("window") or {}
    horizon = int(window.get("slots", 72))
    if window.get("slot_minutes"):
        horizon = int(round(horizon * window["slot_minutes"] / 60.0))
    horizon = min(max(horizon, 1), 72 * 4)
    params = estimate_params(snaps)
    counts, total_slots, n_games = simulate(params, places, horizon)
    forecast = build_forecast(counts, total_slots, n_games, places,
                              params["last_chart"])
    forecast = blend_persist(forecast, params["last_chart"], places)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
