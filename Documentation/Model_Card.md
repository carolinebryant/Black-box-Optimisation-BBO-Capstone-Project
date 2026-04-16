# Model Card — BBO Capstone Optimisation Suite

## Overview

- **Name:** Per-Function Surrogate-Based Optimisation Suite (BBO Capstone)
- **Type:** Sequential model-based optimisation. Eight independent per-function strategies, each built around a surrogate model (which learns the shape of the black-box function from observed points) and an acquisition function (which scores candidate inputs to pick the next query).
- **Version:** v10 (round 10 of a 13-round budget).

## Intended Use

**Suitable for:** Imperial College London and Emeritus Business School professional certificate in Artificial Intelligence and Machine Learning certificate BBO capstone project. Modelling a surrogate on provided initial data to estimate eight unknown black-box functions (ranging from 2-8 dimensions). The goal is generating one query per round, and maximising the function. Starting from a small dataset (10–40 points) and adding one query per week to dataset, working within a 13-round budget.

**Not suitable for:** Settings with different query budgets, batch querying, or substantially different function structure, as all design choices were tuned to this specific sequential, low-data regime. The surrogate choices, kernel settings, exploration weights, and candidate-pool shapes were built around these eight specific functions and their observed trajectories for this specific capstone project. It is not intended for real-world or production use. 

## Details — Per-Function Strategy

Each function was treated separately. The strategy for each one was driven by the function description, per-round observations and data, and what they suggested about the underlying structure of that specific function. Each surrogate was chosen to match that structure, and adjusted accordingly.

**Decision process.**  
At each round, candidate inputs are evaluated using a surrogate model and an acquisition function (UCB or EI). The surrogate provides predicted mean and uncertainty, and the acquisition function balances exploitation and exploration. In some cases, this process is staged (e.g. filtering followed by refinement).

**Function 1 — Radiation Detection (2D).** The function description says only proximity to a source yields a non-zero reading, so most of the input space is a "dead zone". Raw outputs span ~10⁻¹⁰⁷ to ~1, which the surrogate can fit directly. The strategy uses a log-abs transform to convert detection strength into a readable scale, then sets a floor at the largest gap between sorted log readings to stop the GP's uncertainty exploding in the dead zone. Surrogate is an anisotropic Matérn GP with length scale set from the observed distance between two confirmed points on the same source (×1.5, so the GP connects them without merging with other sources).

**Function 2 — Noisy Log-Likelihood (2D).** Described as havibg noisy outputs with multiple local optima. A Matérn GP handles the smoothness; a WhiteKernel term absorbs the noise. Length-scale bounds are derived from partial dependence plots, specifically the input span over which the PDP exceeds 20% of its range; so the kernel is constrained to plausible scales rather than overfitting. UCB with a lowered kappa (1.5 at W9) balances the noise against local-optima exploration.

**Function 3 — Drug Discovery (3D).** Outputs are negative side-effects counts with many near-zero values clustered close to the optimum. A plain GP struggled with this flat region. An ExtraTrees regressor on degree-2 polynomial features performed better, allowing interaction terms (A·B, B·C, etc.), which matched the pattern seen in partial dependence plots. A 50,000-point candidate pool with an exclusion filter (distance > 0.01 from known points) avoids repeated queries.

**Function 4 — Warehouse Placement (4D).** This function is dynamic with many local optima per the description. The GP with anisotropic Matérn kernel was abandoned mid-project after the ARD length scale for x3 hit its upper bound, which signals the GP was treating x3 as globally unimportant. However, local observations showed x3 was highly sensitive near the peak. A single length scale cannot represent both local and global strcture. Switched to Random Forest (LOO R² 0.74 vs GP 0.73), but returned to GP with tighter length-scale bounds at W9 once enough points anchored the landscape.

**Function 5 — Chemical Yield (4D).** Description says typically unimodal. The trajectory showed consistent improvement: eight consecutive improving rounds with x3 and x4 drifting monotonically toward the [0,1] upper bound. Outputs span 2,200 to 6,000, so the GP fits on `log1p(y)` for numerical stability. An SVR acts as a pre-filter (top 5,000 of 10,000 candidates) before the GP+EI stage runs. x3 and x4 are pinned at 0.999999 in candidate generation because the data made the edge-optimum clear; x1 perturbations are biased upward for the same reason.

**Function 6 — Cake Recipe (5D).** The output is a sum of five independent penalty factors (e.g. flavour, consistency, calories, waste, cost), each smooth in its ingredient. This smooth-additive structure suits an SVR with RBF kernel. Hyperparameters (γ=0.05, C=10, ε=0.05) were chosen via LOO grid search over 45 combinations, scored on both global MAE and error at the top-3 observed points. A Random Forest is kept alongside, not for acquisition, but to produce SHAP summary plots for interpretability.

**Function 7 — ML Hyperparameters (6D).** Is simulating a ML model by tuning six hyperparameters. HP1 exhibits feature dominance, hiding the interactions of the other 6 hyperparamteres. Howewver, when HP1 is above HP1≈0.08, there is more noticiable interactions. A single model could not capture both this global constraint and local structure. The strategy uses two Random Forests: a full-dataset RF to screen out obviously bad candidates, and a filtered RF (HP1 < 0.08 subset) to discriminate on HP2–HP6 within the promising region. Permutation importance is checked against impurity importance on both RFs to guard against overfitting on the small filtered subset.

**Function 8 — 8D Black-Box.** 8D ML model with eight hyperparameters, with only 40 seed points is severely under-sampled, thus a GP is underdetermined. An MC-Dropout MLP (hidden=24, dropout=0.25) is used instead, and dropout acts as a cheap Bayesian approximation for uncertainty, and the small network resists overfitting. SHAP values scale the perturbation noise per dimension (active dims get larger jitter). The UCB exploration weight is made adaptive: kappa = kappa_base × (LOO mean error / LOO mean sigma), which closes the feedback loop so the model's own miscalibration determines how much it explores.



## Performance

**Metrics used each round:**

- **Best-so-far progress**: highest output seen across all observations so far, tracked round by round.
- **Leave-one-out (LOO) cross-validation**: used on all eight functions to compare candidate surrogates. Reported as R² and MAE on held-out points.
- **LOO calibration ratio** (mean absolute error / mean predicted standard deviation): used on all eight functions where the surrogate returns an uncertainty estimate.
- **Acquisition diagnostics**: predicted mean, predicted standard deviation, and the acquisition score (UCB or EI) at the chosen query, recorded each round.
- **Cross-surrogate sanity checks**: predictions from two surrogates compared at historical query points before committing a new query.
- **Partial Dependence Plots**: to show the marginal effect of one or two features on a machine learning model's predicted outcome

**What is not reported here:** The raw output values of individual queries. Output magnitudes differ wildly across the eight functions (F1 near 10⁻¹⁰⁷ at W1, F5 around 10³), so they cannot be meaningfully compared on one scale. Per-query values live in the accompanying dataset and datasheet. LOO metrics were used to compare models and track improvement trends, rather than as precise estimates of predictive performance, due to their instability on small sample sizes.

**General pattern across the ten rounds:**

- **F1 (Radiation Detection):** No signal until late (W8: 1.35). Performance improved after introducing log-scaling and a gap-based floor to handle the dead zone.
- **F2 (Noisy Likelihood):** Steady improvement under noise (W2: 0.57 to W8: 0.68). Gains came primarily from tighter length-scale bounds rather than changing the surrogate.
- **F3 (Drug Discovery):** Rapid convergence near optimum (≈0 from W3; best −0.00006 at W6). Performance improved significantly after switching to polynomial features with ExtraTrees.
- **F4 (Warehouse Placement):** Early peak (W2: 0.63) not surpassed. Later rounds focused on local refinement, suggesting non-stationarity and multiple local optima.
- **F5 (Chemical Yield):** Consistent monotonic improvement (2202 to 6014 by W8; 8343 by W9). Strong alignment with the assumed unimodal structure.
- **F6 (Cake Recipe):** Unstable trajectory (−0.83 to −0.29 range). Late-stage SVR hyperparameter tuning (W9) was introduced to address inconsistency.
- **F7 (ML Hyperparameters):** Major improvement at W8 (2.56). Performance increased after introducing the two-stage Random Forest filtering approach.
- **F8 (8D Black-Box):** No improvement beyond initial value (W1: 9.67). Reflects the difficulty of high-dimensional optimisation with limited data.

## Assumptions and Limitations

**Assumptions:**

- Each function is modelling a synthetic function, and not intended for real-life use. Nor is the information extracted rom the data useful or intended to reflect real-world inferences or insights.
- Each input returns a single true output, with no noise (except Function 2, which has noisy outputs and is handled with an explicit noise kernel).
- Inputs live in the unit hypercube [0, 1] per dimension.
- A 13-round budget is enough to find and refine a good region once the general area is identified, but not enough to explore a high-dimensional space from scratch.
- Cross-validation scores on 20–40 points are reliable enough to choose between surrogates, despite the small-sample noise.

**Limitations:**

- **Black-box nature.** The underlying functions are opaque by design, and are not accessable. Imperial College London Computing department owns these functions and provided this data. Every decision is based on patterns inferred from a small set of observations, not on knowledge of the true function. A deceptive landscape, where the best observed region is not the globally best region, will fool this approach.
- **Portability to other projects.** The surrogate choices, kernel settings, candidate-pool shapes, and dimension-pinning decisions were tailored to the specific observed trajectories on these eight functions which are not provided to the public nor participants of this black box challenge. Copying any of them to a new optimisation problem would likely not transfer to other functions.
- **Use in real-life scenarios.** The approach has only attempted to model eight unknown functions, which are unavailable to the participants of this captone challenge. Every week participants submit one query to a portal owned by ICL, and received one output returned within 1 week. Real labs, production systems, or clinical settings involve query cost, measurement error, equipment drift, regulatory constraints, and ethical review. None of which this approach accounts for. The "drug discovery", "chemical yield", and "hyperparameter tuning" labels on the functions are illustrative; this approach has not been validated on any real instance of them.
- **Greedy refinement.** Once a dimension is pinned or a region is committed to (F5's x3=x4=1.0, F7's HP1<0.08), remaining rounds rarely test whether that commitment was premature.
- **Small-sample cross-validation.** LOO scores on 20–40 points move noticeably when a single point is added or removed. They are directional, not definitive.
- Surrogate miscalibration can lead to over- or under-exploration, particularly for GP uncertainty in sparse regions and MC-dropout in high dimensions.
- Acquisition functions (UCB/EI) may exploit misleading uncertainty estimates, reinforcing suboptimal regions.
- Filtering strategies (e.g. F7) risk overfitting to small subsets and excluding globally optimal regions.

## Ethical Considerations

The functions are simulated benchmarks. There is no direct risk of physical, financial, or personal harm. Transparency still matters for two reasons:

1. **Reproducibility.** All experiments were implemented in Python using standard ML libraries (e.g. scikit-learn). Candidate pool sizes ranged from 10,000–50,000 depending on function. Surrogate selection, kernel bounds, and acquisition parameters are documented per function to allow reproduction of decisions. Documenting per-function surrogate choices, diagnostics that drove them, and the reasoning behind surrogate changes are assessable in-line within README. 
2. **Transparency.** This capstone making assumptions regarding what each function is intended to represent based the given function descriptions provided and are not intended to be extended to real BBO problems of a similar description. Anyone wanting to reuse ideas from it on a real problem can check the Assumptions and Limitations section against their setting before any harm is done.

## Caveats and Recommendations

- This card only briefly describes the approach to earlier rounds 1-9, but does provide more in-depth approach as of round 10 of a 13-round budget. Future iterations will add more history relevant for the decision-making process and surrgoate details for weeks 1-9.
- The strategy is still active; W11–W13 may change surrogate or acquisition choices on some functions, and the card will be updated accordingly. 

