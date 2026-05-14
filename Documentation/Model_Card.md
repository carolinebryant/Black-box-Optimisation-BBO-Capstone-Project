# Model Card — BBO Capstone Optimisation Suite

## Overview

- **Name:** Per-Function Surrogate-Based Optimisation Suite (BBO Capstone)
- **Type:** Sequential model-based optimisation. Eight independent per-function strategies, each built around a surrogate model (which learns the shape of the black-box function from observed points) and an acquisition function (which scores candidate inputs to pick the next query).
- **Version:** v13

## Model Description

**Input:** Per-function real-valued vectors ranging from 2D to 8D, with each dimension typically in [0, 1] (Function 5 extends to [0, 5] based on observed data). Inputs are submitted as hyphen-separated strings to ICL's evaluation server.

**Output:** A single scalar value per function per query, representing the black-box function's evaluation at that input. Output scales vary widely across functions (e.g. Function 1 ranges from near-zero to ~1.9, Function 5 from ~2,200 to ~73,500).

## Model Architecture and Strategy

Eight independent surrogate models, one per function. Surrogate classes used include Gaussian Processes, ExtraTrees, SVR, Random Forests, a Residual GP, and an MC-Dropout MLP — each paired with an acquisition function (EI, UCB, LCB, or argmax). Per-function strategy details and weekly performance trends are documented in the [Datasheet](https://github.com/carolinebryant/Black-box-Optimisation-BBO-Capstone-Project/blob/main/Documentation/Datasheet.md).

## Intended Use

**Suitable for:** Imperial College London and Emeritus Business School professional certificate in Artificial Intelligence and Machine Learning certificate BBO capstone project. Modelling a surrogate on provided initial data to estimate eight unknown black-box functions (ranging from 2-8 dimensions). The goal is generating one query per round, and maximising the function. Starting from a small dataset (10–40 points) and adding one query per week to dataset, working within a 13-round budget.

**Not suitable for:** Settings with different query budgets, batch querying, or substantially different function structure, as all design choices were tuned to this specific sequential, low-data regime. The surrogate choices, kernel settings, exploration weights, and candidate-pool shapes were built around these eight specific functions and their observed trajectories for this specific capstone project. It is not intended for real-world or production use. 

## Details — Per-Function Strategy

Each function was treated separately. The strategy for each one was driven by the function description, per-round observations and data, and what they suggested about the underlying structure of that specific function. Each surrogate was chosen to match that structure, and adjusted accordingly. Surrogate choice and tuning evolved over 13 weeks based on per-round diagnostics (LOO calibration, SHAP, partial dependence plots, residual analysis). The descriptions below reflect the final approach for each function.

**Decision process.**  
At each round, candidate inputs are evaluated using a surrogate model and an acquisition function (e.g. UCB, EI, or argmax). The surrogate provides predicted mean and uncertainty, and the acquisition function balances exploitation and exploration. In some cases, this process is staged (e.g. filtering followed by refinement).

**Function 1 — Radiation Detection (2D).** Most of the input space is a dead zone where outputs are near-zero; only proximity to a source yields a non-zero reading. A piecewise scaling transform maps outputs into three regimes: dead zone (abs(y) < 1e-6) to 0.0, weak signal to [1, 2], strong signal to [2, 10]. This replaced an earlier log-abs transform, which compressed the top signal points to the same value and blinded the GP to differences between them. The surrogate is a Matern GP with length scale fixed at the mean pairwise distance of the tightest sub-cluster of confirmed source points (W8/W11/W12). A separate log10 GP is retained for landscape visualisation only. Acquisition is EI with low xi (0.01) for final-week exploitation.

**Function 2 — Noisy Log-Likelihood (2D).** Described as having noisy outputs with multiple local optima. The surrogate is an anisotropic GP with Matern nu=1.5 (rougher than nu=2.5, better suited to noisy data) and a WhiteKernel with raised bounds (0.005-0.2) reflecting LOO residual noise of ~0.18. Per-point heteroscedastic alpha is set at the two duplicate-query points (W8/W11) using their observed discrepancy (variance = 0.018, from abs(W8-W11)/sqrt(2)), encoding known noise into the model. Length-scale bounds (x1: 0.05-1.0, x2: 0.02-0.15) are retained despite looser bounds giving better NLPD, because the tight x2 bound preserves an observed interaction that a stationary ARD kernel would otherwise dismiss. Acquisition is UCB with kappa=0.0 (pure GP mean — full exploitation in the final week).

**Function 3 — Drug Discovery (3D).** Outputs are negative side-effect counts with many near-zero values clustered near the optimum. The surrogate is an ExtraTrees regressor on degree-2 polynomial features with sample weights inversely proportional to abs(Y), which upweights near-zero training points that the unweighted model undervalued. A Gradient Boosting surrogate with Huber loss was tried (W12) but reverted because its piecewise-constant predictions could not rank fine-grained candidates near the optimum. Acquisition is two-stage: LCB (kappa=0.5) selects the top-10 candidates from a 50k pool within a tight bounding box around the top-5 observed points, then the candidate closest to the best observed point (W6) is chosen. An exclusion filter (min distance 0.0005 from training points) prevents duplicate queries.

**Function 4 — Warehouse Placement (4D).** Dynamic function with many local optima. Early ARD GP failed because a single per-dimension length scale could not represent both local sensitivity and global flatness in x3 (the ARD length scale hit its upper bound, treating x3 as globally unimportant despite local observations showing high sensitivity near the peak). The final surrogate is an isotropic Matern GP with fixed length scale = 0.2 and alpha = 1e-3, tuned via LOO calibration grid (calibration improved from 40% at ls=0.4 to 90% at ls=0.2). Candidates are drawn from a tight box around the best observed point (W2, +/-0.02). Acquisition compares pure argmax and UCB (kappa=0.5), selecting whichever candidate has lower GP posterior standard deviation.

**Function 5 — Chemical Yield (4D).** Described as typically unimodal. The trajectory confirmed this: outputs climbed from 2,200 to 73,500 over 12 rounds, with input dimensions drifting monotonically toward and beyond the [0, 1] boundary. Early rounds pinned x3 and x4 at 0.999999 once the edge-optimum became clear; pinning was removed from W10 onward when the data confirmed the space extends beyond [0, 1]. The surrogate is a two-stage pipeline: an SVR (RBF, C=100, epsilon=0.01) pre-filters candidates, then a GP fitted on log1p(y) scores the survivors. Candidates include local perturbations ([0, 5] bounds) and directed outward pushes with per-dimension noise (x1: std=0.4, x2: std=0.3, x3/x4: std=0.2). Acquisition is greedy argmax of GP mean.

**Function 6 — Cake Recipe (5D).** The output is a sum of smooth independent penalty factors (flavour, consistency, calories, waste, cost). The surrogate is an SVR (RBF, C=50, gamma=0.15, epsilon=0.005) with StandardScaler, tuned via a 100-configuration LOO grid search scored on top-5 bias (the mean LOO error at the five best observed points). The selected configuration reduced top-5 bias from +0.011 to +0.002 compared to the previous hyperparameters. Candidate generation is two-stage: Stage 1 draws 5k Maximin LHS points globally, Stage 2 generates 500 local perturbations (+/-0.05) around each of the top-100 Stage 1 candidates. Acquisition is argmax SVR prediction over the combined pool. SHAP runs directly on the SVR for interpretability.

**Function 7 — ML Hyperparameters (6D).** Simulates an ML model with six hyperparameters. HP1 exhibits feature dominance: below HP1 ~ 0.1, outputs are near-zero regardless of HP2-HP6; above it, interactions between the other hyperparameters become visible. The surrogate is a Residual GP: a full-dataset Random Forest (200 trees) provides the base prediction, and a GP (ARD Matern nu=2.5) models the residuals. A filtered RF (HP1 < 0.1 subset) is maintained alongside for SHAP diagnostics and permutation importance validation. Candidates are generated via LHS within bounding boxes around the top-4 observed points, with per-dimension radii set from the spread of those points. Acquisition is EI (xi=0.01) on the combined RF + GP prediction.

**Function 8 — 8D Black-Box.** Eight hyperparameters tuned for an ML model. The surrogate is an MC-Dropout MLP (2 hidden layers, 48 units, SiLU activation, dropout=0.25), where dropout provides a cheap Bayesian uncertainty approximation via 100 forward passes. Training uses weighted MSE (10x for y > 9.5, 5x for y > 9.0) to sharpen predictions near the optimum. The UCB exploration weight is adaptive: kappa = kappa_base x (LOO mean error / LOO mean std), so the model's own miscalibration determines how much it explores. SHAP values on the MLP inform per-dimension understanding. Candidates are 50k global LHS points.



## Performance

**Metrics used each round:**

- **Best-so-far progress**: highest output seen across all observations so far, tracked round by round.
- **Leave-one-out (LOO) cross-validation**: used on all eight functions to compare candidate surrogates. Reported as R² and MAE on held-out points.
- **LOO calibration ratio** (mean absolute error / mean predicted standard deviation): used on all eight functions where the surrogate returns an uncertainty estimate.
- **Acquisition diagnostics**: predicted mean, predicted standard deviation, and the acquisition score (UCB, EI, or argmax) at the chosen query, recorded each round.
- **Cross-surrogate sanity checks**: predictions from two surrogates compared at historical query points before committing a new query.
- **Partial Dependence Plots**: to show the marginal effect of one or two features on a machine learning model's predicted outcome

**What is not reported here:** The raw output values of individual queries. Output magnitudes differ wildly across the eight functions (F1 near 10⁻¹⁰⁷ at W1, F5 around 10³), so they cannot be meaningfully compared on one scale. Per-query values live in the accompanying dataset and datasheet. LOO metrics were used to compare models and track improvement trends, rather than as precise estimates of predictive performance, due to their instability on small sample sizes.

**General pattern across the thirteen rounds:**

- **F1 (Radiation Detection):** No signal for seven consecutive rounds, then a breakthrough at W8 (1.35). W9 dropped sharply (0.08) before recovering through W10-W12 (0.59, 1.14, 1.90). The initial breakthrough came from introducing log-scaling and a gap-based floor to handle the dead zone. Later improvement came from switching to a piecewise transform that preserved magnitude differences between top detections, and tightening the length scale cluster to the three strongest source points.
- **F2 (Noisy Likelihood):** Volatile throughout, peaking twice at W4 and W8 (both 0.68) with a crash at W6 (0.10). W9-W12 settled in the 0.49-0.59 range but never exceeded the W4/W8 peaks. The surrogate class remained an ARD GP throughout; gains came from tighter length-scale bounds, per-point noise estimates from duplicate-query discrepancies, and switching to Matern nu=1.5 (a rougher kernel better suited to noisy data). The volatility likely reflects the function's inherent noise rather than surrogate instability.
- **F3 (Drug Discovery):** Rapid convergence near optimum from W3, reaching best at W6 (-0.00006). However, several regressions occurred (W7: -0.09, W10: -0.03), suggesting the surrogate struggled to maintain precision near zero. Performance improved significantly after switching to polynomial features with ExtraTrees. Inverse-output sample weighting was added later to upweight near-zero training points, and W11 recovered to -0.0005 after W10's regression.
- **F4 (Warehouse Placement):** Early peak (W2: 0.63) never surpassed. W3-W8 produced consistently negative results (-2.46 to -0.99), indicating queries landing in poor local optima. W9-W12 recovered to positive territory (0.19-0.32) after replacing ARD GP with isotropic GP and LOO-tuned length scale (0.4 to 0.2), which improved prediction coverage from 40% to 90%, though the landscape's local optima prevented matching W2.
- **F5 (Chemical Yield):** Consistent monotonic improvement across all twelve rounds (2,202 to 73,499), with acceleration from W10 onward. Strong alignment with the assumed unimodal structure. Candidate bounds expanded beyond [0, 1] from W10 once data confirmed the optimum lies outside the original domain, driving the acceleration in later rounds.
- **F6 (Cake Recipe):** Unstable trajectory (−0.83 to −0.29 range). SVR was introduced at W8 to address inconsistency. A LOO grid search scored on top-5 bias further refined the hyperparameters, yielding the best result at W12 (-0.124).
- **F7 (ML Hyperparameters):** Early peak at W3 (2.60) followed by a collapse at W5 (0.38), then gradual recovery. Major improvement at W8 (2.56). Performance increased after introducing the two-stage Random Forest filtering approach. Then later introduced Residual GP (RF base + GP on residuals) in W11, which better captured structure in the HP1 > 0.1 region. W12 dipped slightly (2.62) but remained competitive.
- **F8 (8D Black-Box):** Results stayed between 9.55 and 9.92 across all rounds, except for a dip at W5 (9.15). Best result at W10 (9.92), with W12 (9.85) recovering after a slight drop at W11 (9.77). The MC-Dropout MLP with adaptive kappa provided stable predictions; weighted MSE training and LOO calibration ratio kept exploration appropriately scaled.

## Assumptions and Limitations

**Assumptions:**

- Each function is modelling a synthetic function, and not intended for real-life use. Nor is the information extracted from the data useful or intended to reflect real-world inferences or insights.
- Each input returns a single true output, with no noise (except Function 2, which has noisy outputs and is handled with an explicit noise kernel).
- Inputs live in the unit hypercube [0, 1] per dimension (except for function 5)
- A 13-round budget is enough to find and refine a good region once the general area is identified, but not enough to explore a high-dimensional space from scratch.
- Cross-validation scores on 20–40 points are reliable enough to choose between surrogates, despite the small-sample noise.

**Limitations:**

- **Black-box nature.** The underlying functions are opaque by design, and are not accessible. Imperial College London Computing department owns these functions and provided this data. Every decision is based on patterns inferred from a small set of observations, not on knowledge of the true function. A deceptive landscape, where the best observed region is not the globally best region, will fool this approach.
- **Portability to other projects.** The surrogate choices, kernel settings, candidate-pool shapes, and dimension-pinning decisions were tailored to the specific observed trajectories on these eight functions which are not provided to the public nor participants of this black box challenge. Copying any of them to a new optimisation problem would likely not transfer to other functions.
- **Use in real-life scenarios.** The approach has only attempted to model eight unknown functions, which are unavailable to the participants of this capstone challenge. Every week participants submit one query to a portal owned by ICL, and received one output returned within 1 week. Real labs, production systems, or clinical settings involve query cost, measurement error, equipment drift, regulatory constraints, and ethical review. None of which this approach accounts for. The "drug discovery", "chemical yield", and "hyperparameter tuning" labels on the functions are illustrative; this approach has not been validated on any real instance of them.
- **Greedy refinement.** Once a dimension is pinned or a region is committed to (F5's x3=x4=1.0, F7's HP1<0.01), remaining rounds rarely test whether that commitment was premature.
- **Small-sample cross-validation.** LOO scores on 20–40 points move noticeably when a single point is added or removed. They are directional, not definitive.
- Surrogate miscalibration can lead to over- or under-exploration, particularly for GP uncertainty in sparse regions and MC-dropout in high dimensions.
- Acquisition functions (UCB/EI) may exploit misleading uncertainty estimates, reinforcing suboptimal regions.
- Filtering strategies (e.g. F7) risk overfitting to small subsets and excluding globally optimal regions.

## Trade-offs

- **GP vs tree-based surrogates.** GPs provide calibrated uncertainty estimates for acquisition functions but struggle with non-stationarity (F4) and flat regions (F3). Tree-based models (ExtraTrees, RF) handle non-linearity better but lack native uncertainty, requiring workarounds like ensemble variance or pairing with a GP.
- **Exploitation vs exploration.** Late-round kappa and xi values were reduced toward zero on several functions (F1, F2, F4) to exploit known good regions. This risks missing undiscovered optima but is appropriate given the limited remaining budget.
- **SVR as pre-filter vs standalone surrogate.** SVR provides fast, stable predictions for candidate filtering (F5, F6) but returns no uncertainty estimate. Where used as a pre-filter (F5), a GP scores the survivors; where used standalone (F6), acquisition is pure argmax with no exploration term.
- **Fixed vs learned length scales.** Learned (ARD) length scales adapt to the data but can overfit with few observations (F4's x3 hitting the upper bound). Fixed length scales tuned via LOO grid search proved more stable on small datasets but require manual retuning as data accumulates.
- **Weighted vs unweighted loss.** Weighting training loss toward high-value points (F8) or near-zero points (F3) improves surrogate accuracy where it matters most, at the cost of worse predictions elsewhere in the input space.


## Ethical Considerations

The functions are simulated benchmarks. There is no direct risk of physical, financial, or personal harm. Transparency still matters for two reasons:

1. **Reproducibility.** All experiments were implemented in Python using standard ML libraries (e.g. scikit-learn). Candidate pool sizes ranged from 10,000–50,000 depending on function. Surrogate selection, kernel bounds, and acquisition parameters are documented per function to allow reproduction of decisions. Documenting per-function surrogate choices, diagnostics that drove them, and the reasoning behind surrogate changes are assessable in-line within README. 
2. **Transparency.** This capstone making assumptions regarding what each function is intended to represent based the given function descriptions provided and are not intended to be extended to real BBO problems of a similar description. Anyone wanting to reuse ideas from it on a real problem can check the Assumptions and Limitations section against their setting before any harm is done.

## Caveats and Recommendations

- This card only briefly describes the approach to earlier rounds 1-9, but does provide more in-depth approach as of round 10 of a 13-round budget. It includes round 13's strategy, but excluding round 13 outputs which are TBD.
- The strategy is still active; thus if changes surrogate or acquisition are made, and the card will be updated accordingly. 

