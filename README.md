# Black-box-Optimisation-BBO-Capstone-Project
Black-box optimisation (BBO) capstone project for Imperial College London and Emeritus Business School professional certificate in Artificial Intelligence and Machine Learning. Maximising eight synthetic black-box functions, using ML techniques in Python.

My repository contains five weekly iteration notebooks (week1_bbo.ipynb through week5_bbo.ipynb), a bbo_data/ directory with per-function initial_inputs.npy and initial_outputs.npy files, and a Function_Descriptions reference document. Weekly results are tracked as hardcoded arrays that accumulate across notebooks.

The main libraries used are scikit-learn, scipy, numpy, and PyTorch. Specifically:

- scikit-learn: Gaussian Processes, Random Forests, SVR, MLP, and polynomial regression as surrogate models
- scipy: Expected Improvement and UCB acquisition functions, Latin Hypercube Sampling for candidate generation
- numpy: data handling and array operations throughout
- PyTorch: introduced in week 5 for a Monte Carlo dropout MLP, where uncertainty estimation was needed beyond what scikit-learn's built-in models could offer.

The README currently describes the project at a high level but doesn't capture the surrogate modelling approach, acquisition strategy, or the reasoning behind key design decisions. Please see the Model Card and Datasheet for more information.
