## Bio–Quantum Hybrid Linear Regression (script + CI)

This repo contains a Python script extracted from the notebook:
`2_EN_Bio_Quantum_Hybrid_Linear_Regression_A_Novel_Approach_Combining_Brain_Organoids_Intelligence_and_Quantum_Computing.ipynb`.

### Run locally

```bash
python -m pip install -r requirements.txt
python run_experiments.py
```

All experiment outputs (text, numbers, plots) are exported into the `results/` folder.

### Run in GitHub Actions

Workflow: `.github/workflows/run-experiments.yml`

After the workflow finishes, download the `results` artifact (`results.zip`) from the run page.

## Current Limitations

This research has several limitations that can be further improved:

1. The quantum computing and organoid intelligence components are still relatively simple in their current implementation.
2. The evaluation is currently limited and does not yet provide strong empirical validation.
3. The dataset used is synthetic and includes only a single feature.
