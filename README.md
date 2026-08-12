# Centralized Federated Learning for Child Malnutrition Prediction

Adapted from your original diabetes-prediction federated demo, rebuilt on
your `malnutrition_child_dataset_cleaned_v3.csv` dataset (198,801 rows).

## What changed from the original diabetes version

| | Original | This version |
|---|---|---|
| Dataset | `pima-indians-diabetes.csv` | `malnutrition_child_dataset_cleaned_v3.csv` |
| Target | `Outcome` (diabetic / not) | `malnourished` (engineered, see below) |
| Clients | 8 near-duplicate scripts (`remoteclient2-5.py`, `subordinateclient2-4.py`) | 1 reusable `client.py`, parameterized by `--client_id` |
| Model | `GradientBoostingClassifier` | Same |
| Aggregation | Server averages client hyperparameter configs (hashed + Laplace-noised) each round | Same mechanism, now a running weighted average across rounds so it converges instead of oscillating |
| Evaluation | `test/compare.py` scored one client's report vs. one reference CSV | `test/compare.py` scores **every** client plus a majority-vote **federated ensemble** across all of them |

## How the malnutrition label was built

Your dataset didn't have a ready-made label column, but it does have the three
WHO child-growth z-scores (`stunting_zscore`, `wasting_zscore`,
`underweight_zscore`). Following the standard WHO/DHS convention, a child is
labeled `malnourished = 1` if **any** of those three z-scores falls below
**-2** (moderate-to-severe malnutrition), else `0`. This gives a nicely
balanced target (~51.7% malnourished / 48.3% not).

Those three z-score columns (and `child_bmi_raw`, which is derived from the
same measurements) are then **dropped from the feature set** — since they
define the label, leaving them in would let the model "cheat" by reading the
answer directly instead of learning from the actual risk factors (age,
weight/height, breastfeeding, illness history, maternal and household
factors, water/sanitation access, etc.).

## Project structure

```
malnutrition_fl/
├── data/
│   └── malnutrition_child_dataset_cleaned_v3.csv   # your raw dataset
├── data_prep.py       # engineers the label, splits into 4 client shards + test set
├── server.py          # federated aggregation server
├── client.py          # reusable client (run once per simulated site)
├── test/
│   └── compare.py     # scores each client + the federated ensemble
├── requirements.txt
└── README.md
```

## How to run it

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Prepare the data** (creates `data/client_1.csv` … `client_4.csv` and `data/test_holdout.csv`)
```bash
python data_prep.py
```

**3. Start the server** (in its own terminal — it waits for 4 clients, then exits)
```bash
python server.py
```

**4. Run each client** (in 4 separate terminals, or one after another — each
one trains locally, federates with the server, and scores the shared test set)
```bash
python client.py --client_id 1 --data data/client_1.csv
python client.py --client_id 2 --data data/client_2.csv
python client.py --client_id 3 --data data/client_3.csv
python client.py --client_id 4 --data data/client_4.csv
```

Each run writes `report_client_<id>.csv` and a folder of diagnostic plots
(`plots/client_<id>/`): feature importance, correlation heatmap, calibration
curve, confusion matrix, SHAP summary, and top-feature distributions by
outcome.

**5. Compare results**
```bash
python test/compare.py
```
This scores every client's report against the true labels in
`data/test_holdout.csv`, plus a **federated ensemble** (majority vote across
all 4 clients' post-federation models), and writes `plots/summary_metrics.csv`.

## Results from a test run

| Client | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Client 1 | 0.9155 | 0.9162 | 0.9206 | 0.9184 |
| Client 2 | 0.9130 | 0.9107 | 0.9221 | 0.9164 |
| Client 3 | 0.9140 | 0.9148 | 0.9192 | 0.9170 |
| Client 4 | 0.9130 | 0.9152 | 0.9166 | 0.9159 |
| **Federated ensemble** | **0.9172** | 0.9076 | **0.9351** | **0.9211** |

The federated ensemble edges out every individual client — the point of the
whole exercise: each site only sees its own local slice of data, but
combining what they learn (without ever pooling the raw records) gives a
better model than any one site could train alone.

## For your viva / report

- **Why federated learning here**: child health data is sensitive and often
  siloed across clinics/health-worker teams; FL lets multiple sites
  collaboratively improve a shared model without centralizing raw patient
  records.
- **Privacy mechanism used**: column names are SHA-256 hashed and numeric
  parameter values get Laplace noise added (differential privacy) before
  leaving a client, mirroring your original design.
- **A caveat worth mentioning if asked**: this demo "federates" by averaging
  `GradientBoostingClassifier` *hyperparameters/config* across clients
  (matching your original project's design), not the tree ensemble weights
  themselves — true parameter-level FedAvg is more natural for models with a
  fixed-size weight vector (e.g. logistic regression or a neural net). If
  your evaluator pushes on this, you can frame it as: "the trees stay local;
  what's federated is the shared training configuration each site's local
  model gets refit against."
