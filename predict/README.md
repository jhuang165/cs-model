# CS2 match prediction experiments

Standalone Python package (no changes to Valve's JS model) that trains and backtests
online rating models on `data/matchdata_sample_20230829.json` (7,881 series with
5-man rosters, 2022-08-29 to 2023-08-29, map scores for every series).

```
python3 -m venv .venv && .venv/bin/pip install numpy pandas scikit-learn scipy matplotlib
.venv/bin/python -m predict.run_backtest --plot predict/calibration.png   # compare models
.venv/bin/python -m predict.run_backtest --min-history 5                   # warm lineups only
.venv/bin/python -m predict.rankings --top 30                              # current ratings
 .venv/bin/python -m predict.rankings --vs Vitality FaZe --bo 3             # head-to-head price
.venv/bin/python -m predict.batch --tau 180,365 --C 3,10 --scale           # batch Bradley-Terry sweep
.venv/bin/python -m predict.valve_baseline                                 # score Valve's own model
```

## Layout

| file | purpose |
|---|---|
| `data.py` | loads the JSON into time-ordered `Match` records; infers BO1/BO3/BO5 from maps the winner took |
| `models.py` | online models: constant, team1 base rate, team-id Elo, player-mean Elo (series / per-map / round-share), player Glicko, player Glicko + per-map offsets, online bias wrapper |
| `evaluate.py` | walk-forward loop (predict before update, always), log loss, Brier, accuracy, AUC, calibration, ECE |
| `run_backtest.py` | runs the model zoo, prints a table and calibration bins |
| `run_maps.py` | map-aware models scored at series level and per individual map (`--sweep` for k_map/shrink grid) |
| `rankings.py` | fits the shipped model (regional Glicko + batch blend + temperature) on everything and prints current lineups' ratings or a matchup probability |
| `sweep.py` | grid-search Glicko parameters on a tuning window, confirm on a later held-out window |
| `stack.py` | stacked logistic / boosting layer over walk-forward features, incl. a joint refit of the shipped blend weight and temperature |
| `regions.py` | country -> region lookup read from Valve's `model/util/region.js` (plus a CIS split) |
| `batch.py` | time-weighted batch Bradley-Terry with a regional prior, refit daily, evaluated walk-forward |
| `valve_baseline.py` / `.js` | runs Valve's own `model/ranking.js` weekly and scores it on the same matches and metrics |
| `pandascore_export.py` | pulls CS matches from the PandaScore free tier into Valve's schema (`data/matchdata_pandascore.json`) |

Every runner takes `--data <file>` to point at a different Valve-schema file.

## Backtest (metrics on matches from 2023-03-01, models warmed on the prior 6 months)

| model | logloss | brier | acc | auc | ece |
|---|---|---|---|---|---|
| constant 0.5 | 0.693 | 0.250 | 0.573 | 0.500 | 0.073 |
| team1 base rate | 0.683 | 0.245 | 0.573 | 0.473 | 0.016 |
| team-id Elo, k=40 | 0.649 | 0.229 | 0.615 | 0.655 | 0.044 |
| player Elo, series update, k=40 | 0.641 | 0.225 | 0.634 | 0.671 | 0.044 |
| player Elo, per-map update, k=30 | 0.634 | 0.222 | 0.636 | 0.682 | 0.035 |
| player Elo, per-map + round share, k=30 | 0.635 | 0.222 | 0.640 | 0.692 | 0.049 |
| player Glicko, per-map, rd0=200 c=20 | **0.625** | **0.218** | **0.654** | 0.695 | 0.030 |
| ... + online team1 bias | 0.624 | 0.218 | 0.645 | 0.695 | 0.021 |

Takeaways so far:

- **Rate players, not team ids.** Team rating = mean of the five players. Handles stand-ins and
  roster moves for free and beats team-id Elo on every metric.
- **Update per map, predict the series.** A 2-1 BO3 is three observations, not one. Series
  probability comes from the per-map probability through the BO formula.
- **Uncertainty helps.** Glicko's rating deviation lets new players move fast and veterans move
  slowly; it is the single biggest gain after going player-level. Lower starting RD (200) and
  slower RD growth (c=20) beat Valve's defaults (350 / 34.6) for prediction.
- **Round share adds AUC but hurts calibration** in the naive form here. Worth revisiting with a
  proper margin model rather than a stretched blend.
- **team1 is listed first non-randomly** (wins 56%). With `--min-history 5` the base rate's ECE
  drops to 0 and the bias wrapper stops helping, so the effect is almost entirely cold-start
  matches where the unknown team happens to be the weaker one. Do not hard-code it.
- Remaining miscalibration is in the 0.0-0.2 bins (n < 120): heavy underdogs listed as team1 win
  more often than predicted.

## Per-map ratings (negative result)

`PlayerMapGlicko` adds a per-(player, map) offset on top of the global Glicko rating, updated
with step `k_map` and shrunk toward zero each update. With the veto known (maps actually played,
unplayed decider priced map-agnostically) it is scored at series level and per individual map.

| model | series logloss | map logloss | map auc |
|---|---|---|---|
| player Glicko (map-agnostic) | 0.6255 | 0.6545 | 0.645 |
| + map offsets, k_map=10, veto unknown | 0.6250 | 0.6525 | 0.649 |
| + map offsets, k_map=10, veto known | 0.6254 | 0.6525 | 0.649 |
| + map offsets, k_map=30 | 0.6280 | 0.6576 | 0.645 |

The gain is ~0.002 at map level and nil at series level, and larger `k_map` hurts. Two
diagnostics explain why:

- **Map-specific skill does not persist.** For every (teamId, map) with 8+ maps, the mean
  residual (outcome minus Glicko probability) in the first half of its history has correlation
  0.04 with the second half, and 0.00 once the team's overall residual is removed. Whatever map
  edge exists is vetoed away or shifts faster than one season of data can track.
- **Map order carries no pick information.** In BO3s team1 wins map 1 and map 2 at the same
  rate (54.6% vs 54.4%), whether favourite or underdog, so the listed order is not the pick order.

The map model is kept because it is never worse, but it is not where the next gain is.

## Getting more data: PandaScore

```
export PANDASCORE_TOKEN=...   # free token from https://app.pandascore.co
.venv/bin/python -m predict.pandascore_export --since 2023-09-01 --rosters
.venv/bin/python -m predict.run_backtest --data data/matchdata_pandascore.json --eval-from 2025-01-01
```

The raw pull is cached in `data/pandascore/matches_raw.jsonl` and is incremental, so re-running
with a later `--since` only fetches new matches. Free tier limits: 60 requests/minute,
1,000/hour, 100 matches per request; a full backfill from September 2023 is a few hundred requests.

What the free plan gives and withholds, and what that means for the models:

| field | free plan | consequence |
|---|---|---|
| series winner, per-game winner, number_of_games, forfeit | yes | per-map binary updates and BO-aware series pricing work |
| tournament prize pool, tier, region | yes | usable for event weighting / regional prior |
| round scores per map | **no** (Historical plan) | maps are stored 1-0; round-share variants fall back to binary |
| map names | **no** | per-map offsets inert (they were ~useless anyway) |
| per-match lineups | **no** | each team is one synthetic `team:<id>` "player", so player-level models degrade to team-level. `--rosters` snapshots current lineups and applies them to matches after the snapshot only, so player-level ratings accrue going forward |

The PandaScore team/player ids do not match the HLTV ids in Valve's 2023 sample, so keep the two
files separate rather than concatenating them.

### Results on the PandaScore export (45,022 series, Sep 2023 - Sep 2026; scored from 2024-07-01, n=33,489)

| model | logloss | brier | acc | auc | ece |
|---|---|---|---|---|---|
| constant 0.5 | 0.693 | 0.250 | 0.549 | 0.500 | 0.049 |
| team-id Elo, k=40 | 0.639 | 0.224 | 0.632 | 0.677 | 0.020 |
| team Elo, per-map update, k=20 | 0.638 | 0.224 | 0.632 | 0.680 | 0.022 |
| team Glicko, per-map, rd0=150 c=15 | **0.630** | **0.220** | 0.642 | 0.693 | 0.019 |
| ... + online logit scale (a -> 0.83) | 0.629 | 0.220 | 0.642 | 0.696 | 0.016 |

(Player-level models equal team-level ones here because the free plan has no lineups.)

Breakdown of the best model: tier S 0.622 / 63.9% acc (n=956), tier A 0.632, tier C 0.613,
tier D 0.635; by year 2024 0.637, 2025 0.631, 2026 0.622 (ratings keep improving as history
accrues). BO3 (0.628) is easier than BO1 (0.633). The raw model is slightly overconfident at both
extremes; a learned logit temperature of ~0.83 fixes it, so `OnlineScale` is the recommended top layer.

## Parameter sweep and stacking (PandaScore data; tune/train Jul 2024 - Jun 2025, test Jul 2025 - Sep 2026)

**Sweep** (`sweep.py`, 120 configs of rd0 x c x min_rd): the surface is flat. Best on the tuning
window is rd0=150, c=20, and min_rd never binds. Held-out log loss 0.6249 vs 0.6257 for the old
default, so tuning is worth ~0.001. Adding `OnlineScale` on top is worth ~0.003 more (0.6222).

**Stacking** (`stack.py`, 23 features computed walk-forward: base-model logits, rating diff,
both RDs, tier, best-of, log prize pool, and per-team rest days / experience / last-10 form):

| model on the held-out window (n=18,229) | logloss | acc | ece |
|---|---|---|---|
| base Glicko (150, 20, 30) | 0.6249 | 0.649 | 0.029 |
| logistic on base logit only (offline temperature 0.86 + intercept) | 0.6218 | 0.652 | 0.009 |
| **logistic, all features** | **0.6160** | **0.657** | **0.006** |
| gradient boosting, all features | 0.6187 | 0.652 | 0.009 |
| logistic, side features only, no ratings | 0.6548 | 0.609 | 0.007 |

Reading the coefficients: the stacker mostly re-learns the map-to-series conversion (large
weights on the per-map logit and the best-of dummies), then adds small mean-reversion on recent
form (a hot streak means the rating has overshot). Boosting adds nothing over linear.

By tier, the stacking gain is concentrated in tiers C and D (0.005 and 0.011 log loss); tiers
S/A/B improve by 0.001 to 0.006 and on 500 to 700 matches that is within noise. For top-tier
prediction the base model plus a temperature is nearly as good as the full stack.

### Blend weight and temperature fitted jointly

`stack.py` now also carries the shipped model's two halves as features (`z_batch`, `z_glicko`,
series logits) and fits `P = sigmoid(a_b z_batch + a_g z_glicko)` on the training window, i.e. the
blend weight `w = a_b / (a_b + a_g)` and the temperature `a_b + a_g` together, instead of a fixed
0.3 and an online temperature. It also refits monthly on everything before each block (`walk_fit`),
which is what a live refit would see. Valve sample: train Nov 2022 - Feb 2023, test from Mar 2023
(n=3,815); PandaScore: windows as above.

| test log loss | Valve sample | PandaScore |
|---|---|---|
| shipped (0.3 / 0.7 + online temperature) | **0.6146** | **0.6175** |
| joint LR, fitted once (w, a) | 0.6149 (0.37, 0.74) | 0.6178 (0.50, 0.84) |
| joint LR + intercept | 0.6135 | 0.6173 |
| joint LR, refit monthly | 0.6145 | 0.6176 |
| full LR, all features incl. `z_batch`/`z_glicko`, refit monthly | 0.6044 | 0.6117 |

**Negative result.** The loss is flat for w between 0.2 and 0.5 (within 0.0005 on both datasets) once the
temperature is refitted for each w, and the online temperature already tracks the offline optimum.
The shipped weight stays. The only thing a joint fit adds is an intercept (team1 listed first,
0.001), which the README already advises against hard-coding. The full stacker on top of the blend is
the one that still pays (0.010 Valve, 0.006 PandaScore); see next steps.

## Regional prior (`RegionalGlicko`)

Each match now carries player countries (Valve sample) or the team's PandaScore location (put in the
synthetic player's `countryIso`), mapped to nine regions with Valve's own country table plus a CIS
split. `RegionalGlicko` adds two independent pieces to `PlayerGlicko`:

- **seed**: a new player's first rating is the mean rating of every rated player from its region, not 1500.
- **offset**: a per-region rating offset learned online from cross-region matches only. Teams that
  never leave their region are only rated relative to each other; the offset moves the whole region
  when its travellers win or lose abroad.

| model (log loss; rd0=200 on the Valve sample, 150 on PandaScore, c=20) | Valve sample (from 2023-03) | PandaScore (from 2025-07) |
|---|---|---|
| player Glicko | 0.6255 | 0.6249 |
| ... + scale | 0.6252 | 0.6222 |
| regional, seed only | 0.6212 | 0.6227 |
| regional, offset only | 0.6215 | 0.6245 |
| regional, seed + offset | **0.6175** | 0.6231 |
| regional, seed + offset + scale | 0.6180 | **0.6197** |

Notes:

- Seeding from *settled* players only (RD < 80) is worse than the base model (0.6356): settled
  players are the survivors and rate well above what a newcomer ends up at. Seeding from a running
  mean of newcomers' ratings at their 10th map also loses. The plain mean of everyone works.
- Offsets are worth 0.004 on the Valve sample (real rosters, many cross-region LANs) and nothing on
  PandaScore, where the regional offset ends up as CIS +110, OC -62, unknown-location -59.
- Lynn Vision drops out of the top 10 under this model.

## Time-weighted batch fit (`batch.py`)

Bradley-Terry per map, team logit = mean of player logits (+ region effect), fitted as a weighted
ridge logistic regression where map weights decay with half-life `tau`. Refit on a schedule and
evaluated walk-forward like everything else: only maps that started before the match being
predicted are in the fit. A full three-year PandaScore fit takes about 0.1 s, so daily refits are cheap.

| model | Valve sample | PandaScore |
|---|---|---|
| best weekly refit (Valve: tau=365, C=10; PS: tau=180, C=3), region + scale | 0.6199 | 0.6276 |
| same, daily refit | **0.6178** | 0.6224 |
| regional Glicko (+scale on PS) for reference | 0.6175 | 0.6197 |
| blend of the two, 0.3 batch / 0.7 Glicko logits, + scale | **0.6143** | **0.6190** |

Findings:

- Raw batch fits are overconfident at any useful C (ECE 0.03-0.07); the online temperature fixes it.
  Weak ridge (C=30) is bad, strong ridge (C=1 or less) is underconfident; the sweet spot depends on
  whether entities are players (Valve, each carries 1/5 of a team) or teams (PandaScore).
- Refit cadence matters more than any other knob: weekly to daily is worth 0.002 to 0.005. The
  batch fit's disadvantage is staleness, not information.
- Half-life: on the one-year Valve sample longer is better (365 d); on three years of PandaScore
  180 d beats 365 and 730, and 60 to 90 d is clearly too short.
- The region columns help on the Valve sample and are neutral on PandaScore, same as for Glicko.
- Batch and Glicko disagree enough that blending helps: 0.003 on the Valve sample, 0.001 on PandaScore.

## Round scores (`round_weight`, Valve sample only)

PandaScore's free plan stores every map as 1-0, so this only applies to the Valve sample (16,953 maps
with ten or more rounds, all MR15 with overtime). Both halves of the blend get the same margin
likelihood: a map's rounds are treated as Bernoulli trials whose logit is `round_scale` times the map
logit, down-weighted by `round_weight` because rounds within a map are correlated (economy,
momentum). With iid rounds, `s = 0.23` would reproduce the map-win curve of a 30-round map. The
binary map result stays in the likelihood.

- `BatchBT(round_weight=λ, round_scale=s)` adds two weighted rows per map (y=1 with weight λ·t1_rounds,
  y=0 with weight λ·t2_rounds) whose design is scaled by `s`, which is the binomial likelihood.
- `RegionalGlicko(round_weight=λ, round_scale=s)` does a second Glicko-1 update per map with `n = λ·rounds`
  observations at slope `s` and the round share as the score. This is a proper likelihood, unlike the
  stretched round-share blend in `PlayerElo`, which hurt calibration.

Chosen on Mar-May 2023, confirmed on Jun-Aug 2023:

| model (log loss) | tune | confirm | whole window from 2023-03 |
|---|---|---|---|
| batch (tau=365, C=10) + scale | 0.6114 | 0.6271 | 0.6178 |
| ... + rounds, λ=0.5, s=0.25 | 0.6060 | 0.6201 | 0.6118 |
| regional Glicko + scale | 0.6090 | 0.6310 | 0.6180 |
| ... + rounds, λ=0.5, s=0.25 | 0.6009 | 0.6264 | 0.6113 |
| previous shipped blend | 0.6061 | 0.6268 | 0.6146 |
| **blend, batch λ=2 C=3 + Glicko λ=1, s=0.25, w=0.3** | **0.5961** | **0.6216** | **0.6066** |

- Round margins are worth 0.006-0.007 to each half alone and 0.008 to the blend. This is the largest
  single gain since the regional prior.
- Once the rounds carry the information, the batch fit wants a stronger ridge (C=3, not 10) and more round weight.
  Beyond λ=1 the surface is flat to within 0.001, and so is w between 0.3 and 0.5.
- s=0.15 to 0.35 all work; 0.25 was best or tied on both halves, close to the iid value.
- Both halves become a little more confident with rounds (online temperature 0.83 -> 0.72).

## Valve's model as a baseline (`valve_baseline.py`)

Every week Valve's standings are rebuilt with `model/ranking.js` (six-month window, prize and
network seeding, fixed-RD Glicko) from data before that date, and the next week's matches are
priced from the two rosters' rank values with the formula in `model/fit.js`. The PandaScore file is
padded to five copies of the synthetic player, event winners are credited with the prize pool, and
tier S/A events stand in for LAN, otherwise Valve's seeding is NaN. About 78% of matches have both
rosters in the standings; the rest are scored 0.5.

| model, matches where Valve has a standing | Valve sample (n=2981) | PandaScore (n=14031) |
|---|---|---|
| Valve `ranking.js` + `fit.js` expectation | 0.687 / acc 0.637 / auc 0.668 | 0.661 / 0.637 / 0.678 |
| ... + online logit temperature (a -> 0.55) | 0.658 | 0.644 |
| player Glicko | 0.632 / 0.649 / 0.689 | 0.626 / 0.648 / 0.700 |
| regional Glicko + scale | **0.625** / 0.655 / 0.698 | **0.622** / 0.650 / 0.701 |

Valve's expectation curve is far too steep: matches it prices at 95% are won 78-79% of the time,
and at 5% they are won 25-33%. Halving the logit (a=0.55) recovers most of that, but its ranking
information (AUC 0.67-0.68) is still below plain Glicko (0.69-0.70), which is the gap that matters
for the standings use case. The fit.js evaluation uses `Math.max(array)`, which is NaN whenever a
roster matches more than one team; the harness uses the intended maximum.

## Known weaknesses / next steps

1. **Regional isolation.** Done, see above: seed + offset is now the default model.
2. **Stale rosters in `rankings.py`.** Done: a team is listed only if a majority of its latest
   lineup last played for it (Outsiders -> Virtus.pro now shows once), and `--active-days`
   (default 180) hides teams that have stopped playing.
3. **Map pool.** Tried, see above. Would need pick/ban data (which team picked which map) to
   get more than the ~0.002 map-level gain.
4. **Tuning.** Done, see above; flat surface, little to gain.
5. **Stacking.** Done, see above. To use it for live predictions `rankings.py` would need to
   carry the feature extractor and a fitted model; today it only runs the blend. The full LR on top
   of the current shipped model is still worth 0.006 on the Valve sample (0.6003 refit monthly vs
   0.6066) and 0.006 on PandaScore, which makes shipping it the largest remaining gain. Its biggest non-rating
   coefficients are the two RDs with opposite signs (an uncertain team2 favours team1), i.e.
   newcomers are still weaker than the regional seed assumes. A cheaper fix to try first: seed
   below the regional mean, or a learned newcomer offset.
6. **More data.** One season is thin. The Valve JSON schema is the loader's only dependency, so a
   scrape/export in the same shape drops in without code changes.
7. **Ship the blend.** Done, see "Shipped model" below.
8. **Joint blend weight + temperature.** Done, negative: the shipped 0.3 / online temperature sits on the
   flat optimum.
9. **Round scores.** Done and shipped on the Valve sample (0.6146 -> 0.6066). Needs a data source with
   round scores to matter on PandaScore (the paid Historical plan, or a different export).

## Shipped model (`rankings.py`)

`rankings.best_model()` is `OnlineScale(Blend(BatchBT, RegionalGlicko, w=0.3))`, i.e. 0.3 batch /
0.7 Glicko on series logits under an online temperature, with per-dataset settings: on the Valve sample
Glicko rd0=200 with round margins (λ=1), batch tau=365 d, C=3 with round margins (λ=2), s=0.25;
on PandaScore rd0=150, tau=180 d, C=3 and no rounds (picked automatically from whether the rosters
are synthetic `team:` ids). `run_backtest.py` scores the same object as its last entry:

| | Valve sample (from 2023-03) | PandaScore (from 2025-07) |
|---|---|---|
| log loss / acc / auc | 0.6066 / 0.669 / 0.721 | 0.6175 / 0.655 / 0.708 |
| learned temperature | 0.72 | 0.76 |

- **Head-to-heads** (`--vs A B --bo N`) run the full model on a synthetic match between the two
  current lineups, so they carry the BO conversion, region effects and temperature. The implied
  per-map probability is backed out of the series price.
- **The displayed rating** is `1500 + a * (0.3 * batch + 0.7 * Glicko)` on the Elo scale, both halves
  including their region effects. It orders teams the way the model does but is only approximate
  for pricing; use `--vs` for that. The ± is the Glicko team RD.
- A full PandaScore run takes about 90 s (daily batch refits over three years); the Valve sample about
  25 s (the round rows triple the batch fit's size).
