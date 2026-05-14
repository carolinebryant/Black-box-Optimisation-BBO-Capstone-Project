# Black-box-Optimisation-BBO-Capstone-Project
Black-box optimisation (BBO) capstone project for Imperial College London and Emeritus Business School professional certificate in Artificial Intelligence and Machine Learning. The goal is to maximise eight synthetic black-box functions (ranging from 2 to 8 dimensions) where the underlying function is unknown. Each week, one query is submitted per function and a single output is returned. Machine learning surrogate models are trained on the accumulated data to predict the function's shape and guide the next query. The project ran for 13 weekly rounds.

My repository contains one file for each week's iteration of the challenge contained within the /Code folder (week1_bbo.ipynb through week13_bbo.ipynb). The /Data directory contains per-function initial_inputs.npy and initial_outputs.npy files, and a Function Descriptions reference document. Weekly results are tracked as hardcoded arrays that accumulate across notebooks to leave initial data untouched.

The main libraries used are scikit-learn, scipy, numpy, and PyTorch. Specifically:

- scikit-learn: Gaussian Processes, Random Forests, SVR, ExtraTrees, and polynomial regression as surrogate models
- scipy: Expected Improvement and UCB acquisition functions, Latin Hypercube Sampling for candidate generation
- numpy: data handling and array operations throughout
- PyTorch: Monte Carlo dropout MLP for Function 8, where uncertainty estimation was needed beyond what scikit-learn's built-in models could offer

The README currently describes the project at a high level but doesn't capture the surrogate modelling approach, acquisition strategy, or the reasoning behind key design decisions. Please see the [Model Card](/Documentation/Model_Card.md) and [Datasheet](/Documentation/Datasheet.md) for more information on strategy details and results.
