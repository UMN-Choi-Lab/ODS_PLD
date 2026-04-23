# ods-nnls-pld

Analytical OD demand estimation via Non-Negative Least Squares (NNLS) and
Projected Langevin Dynamics (PLD), validated under real SUMO simulations on the
[BO4Mob](https://github.com/UMN-Choi-Lab/BO4Mob) benchmark.


## Method at a glance

Given an assignment matrix `A ∈ R^{m×n}` and observed link counts `y ∈ R^m`:

- **NNLS** returns the analytical minimizer `x* = argmin_{x ≥ 0} ||Ax - y||^2`
  via scipy's BVLS (Lawson–Hanson active-set). One SUMO evaluation validates it.
- **PLD** samples the non-negative surrogate posterior
  `f(x) ∝ exp(-||Ax - y||^2 / τ) · 1[x ≥ 0]`
  via the reflected Langevin update
  `x_{t+1} = max(0, x_t - η · 2Aᵀ(Ax_t - y) + √(2ητ) · z_t)`.
  The chain is initialized at the NNLS mode and the returned sample set
  **includes that mode as `samples[0]`** (controlled by `include_init=True`);
  the remaining `N-1` rows are thinned post-burn-in draws from the chain.
  This makes "PLD best-N ≥ NNLS" a guarantee under any evaluator — not just
  under the surrogate `g^A`.
- **PLD+TuRBO** replaces the Sobol initialization phase of
  [BO4Mob](https://github.com/UMN-Choi-Lab/BO4Mob)'s TuRBO with PLD posterior
  samples (including NNLS as the first init point); the GP model, Thompson
  sampling, and trust region logic are unchanged.

## Layout

```
odspld/
├── networks.py       build A + gt_od + counts from BO4Mob routes/sensor CSVs
├── nnls.py           scipy-BVLS solver + KKT residual
├── pld.py            projected Langevin dynamics
├── turbo.py          PLD-initialized TuRBO (BO4Mob-matched GP/acquisition)
├── baselines.py      Sobol-init best-of-N, SPSA
├── sumo.py           thin wrapper around BO4Mob/src/single_od_run.py
└── paths.py          BO4Mob + SUMO path resolution

scripts/
├── run_nnls.py               run NNLS → SUMO for one network
├── run_pld_bestofN.py        PLD best-of-N → SUMO for one network
├── run_pld_turbo.py          PLD+TuRBO → SUMO for one network
├── reproduce_table.py        orchestrate all networks × seeds
└── collate_results.py        collect JSONs → latex / markdown table

results/   (gitignored) JSON outputs, one per (method, network, seed)
external/  git submodule -> UMN-Choi-Lab/BO4Mob
```

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/UMN-Choi-Lab/ODS_PLD
cd ODS_PLD
# or, if already cloned:
git submodule update --init --recursive
```

### 2. Install SUMO

SUMO is a C++ simulator, not a pip package.

```bash
# Ubuntu / Debian
sudo apt-get install sumo sumo-tools sumo-doc
export SUMO_HOME=/usr/share/sumo

# macOS (Homebrew)
brew tap dlr-ts/sumo && brew install sumo
export SUMO_HOME=/opt/homebrew/share/sumo
```

The `single_od_run.py` entry point inside BO4Mob requires `SUMO_HOME` and the
`sumo` binary on `PATH`.

### 3. Install the package

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

This pulls `numpy`, `scipy`, `torch`, `botorch`, `gpytorch`. BO4Mob itself is
imported by path (the submodule under `external/BO4Mob`); no pip install is
required for it.

### 4. (Optional) point at a pre-existing BO4Mob checkout

If you already have BO4Mob cloned elsewhere:

```bash
export BO4MOB_PATH=/absolute/path/to/BO4Mob
```

## Reproducing the table

### Everything (≈60 hours of SUMO on a workstation)

```bash
python scripts/reproduce_table.py --seeds 11 12 13
python scripts/collate_results.py --format md
```

### One (network, seed) cell

```bash
python scripts/run_nnls.py        --network 2corridor --seed 11
python scripts/run_pld_bestofN.py --network 2corridor --seed 11 --N 20
python scripts/run_pld_turbo.py   --network 2corridor --seed 11
```

Each script writes one JSON under `results/`:

```
results/
├── nnls_2corridor_seed11.json
├── pld_2corridor_seed11.json
└── pldturbo_2corridor_seed11.json
```

Then collate:

```bash
python scripts/collate_results.py --format md
python scripts/collate_results.py --format latex > table.tex
```


## Sweep v1 reproduction (2026-04-20)

| Network          | NNLS (1 eval) | PLD best-N (N=20) | PLD+TuRBO       | Sobol+TuRBO     |
|------------------|---------------|-------------------|-----------------|-----------------|
| 1ramp (n=3)      | 0.0000        | 0.0000            | **0.0000**      | 0.0004          |
| 2corridor (n=21) | 0.1858        | 0.1856            | **0.1778**      | 0.1954*         |
| 3junction (n=44) | 0.3378        | 0.3037            | **0.2081**†     | 0.2799†         |


Key observations:

- **PLD+TuRBO wins on every network.** Breaks the analytical NNLS floor on
  2corridor (0.1778 < 0.1858, −4.3%) and 3junction (0.2081 < 0.3378, −38%) —
  TuRBO exploration recovers nonlinear residual that NNLS's linearized
  assignment matrix can't see.
- **PLD warm-start is the active ingredient.** Sobol+TuRBO is consistently
  worse than PLD+TuRBO and even worse than NNLS on 2corridor (0.1954 > 0.1858).
  100+ GP-optimized SUMO evaluations starting from Sobol cannot match a single
  closed-form NNLS solve. On 3junction the gap is 0.2799 vs 0.2081 (34%
  relative improvement from PLD init at identical TuRBO budget).
- **PLD best-N adds real value on 3junction.** Best-N beats NNLS by 10%
  (0.3037 vs 0.3378) with just N=20 PLD draws. On 1ramp/2corridor the
  `PLD best-N ≥ NNLS` bound is essentially tight; on 3junction it is strict.

Full per-cell numbers and run URLs:
<https://wandb.ai/benchoi93/ODS_PLD?filter=tag%3Asweep_v1>

## Wall-clock budget per seed

| Network | n | SUMO evals | approx. wall clock |
|---|---|---|---|
| 1ramp        | 3   | 10 + 50·2   | ~10 min |
| 2corridor    | 21  | 20 + 100·3  | ~1 hr   |
| 3junction    | 44  | 30 + 200·4  | ~6 hr   |
| 4smallRegion | 151 | 50 + 600·5  | ~40 hr  |

## License

MIT — see [LICENSE](LICENSE).

