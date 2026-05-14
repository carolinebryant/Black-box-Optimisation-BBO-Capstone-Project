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


## Performance

**Metrics used each round:**

- **Best-so-far progress**: highest output seen across all observations so far, tracked round by round.
- **Leave-one-out (LOO) cross-validation**: used on all eight functions to compare candidate surrogates. Reported as R² and MAE on held-out points.
- **LOO calibration ratio** (mean absolute error / mean predicted standard deviation): used on all eight functions where the surrogate returns an uncertainty estimate.
- **Acquisition diagnostics**: predicted mean, predicted standard deviation, and the acquisition score (UCB, EI, or argmax) at the chosen query, recorded each round.
- **Cross-surrogate sanity checks**: predictions from two surrogates compared at historical query points before committing a new query.
- **Partial Dependence Plots**: to show the marginal effect of one or two features on a machine learning model's predicted outcome

**What is not reported here:** The raw output values of individual queries. Output magnitudes differ wildly across the eight functions (F1 near 10⁻¹⁰⁷ at W1, F5 around 10³), so they cannot be meaningfully compared on one scale. Per-query values live in the accompanying dataset and datasheet. LOO metrics were used to compare models and track improvement trends, rather than as precise estimates of predictive performance, due to their instability on small sample sizes.

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
- If changes surrogate or acquisition are made the card will be updated accordingly. 

