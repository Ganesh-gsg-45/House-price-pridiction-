# House Price Prediction

Small ML project + Streamlit UI for house price prediction.

## Project structure
- `data/` — raw CSV data
- `notebook/` — exploratory notebook and scripts
- `src/` — project source code
  - `components/` — ingestion, transformation, training modules
  - `pipeline/` — prediction pipeline
- `artifacts/` — generated artifacts (`train.csv`, `test.csv`, `raw.csv`, `preprocess.pkl`, `model.pkl`)
- `deploy_artifacts/` — optional folder for packaging artifacts for deployment
- `app.py` — Streamlit app
- `pipeline.py` — run full ETL + training pipeline
- `scripts/package_artifacts.py` — package artifacts into `deploy_artifacts/` and zip

---

## Quick Start (local)
1. Create virtual environment and install dependencies:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1  # PowerShell
pip install -r requirements.txt
```

2. Run full pipeline (ingest → transform → train):

```powershell
python -m pipeline
```

This will create `artifacts/` with `train.csv`, `test.csv`, `preprocess.pkl`, and `model.pkl`.

3. Test prediction via script:

```powershell
python -m test_prediction
```

4. Run Streamlit app:

```powershell
streamlit run app.py
```

Open the printed local URL in your browser.

---

## Packaging artifacts for deployment
If you deploy your app to a host that doesn't run the pipeline, include the trained artifacts with the app to avoid runtime training or sklearn-version mismatches.

Create a deployment package (copies `model.pkl` and `preprocess.pkl` into `deploy_artifacts/` and creates a zip):

```powershell
python scripts/package_artifacts.py --zip
```

Upload the `deploy_artifacts/` folder or `deploy_artifacts.zip` with your Streamlit app on the server. The app is configured to prefer `deploy_artifacts/` when present.

**Note:** `deploy_artifacts/` is included in `.gitignore` by default to avoid accidentally committing large files. If you want to commit artifacts, remove that entry from `.gitignore`.

---

## Deployment notes & common issue (sklearn pickle errors)
- Pickles created with one scikit-learn version may not unpickle on a different scikit-learn version. The common error looks like:

```
Can't get attribute '_RemainderColsList' on <module 'sklearn.compose._column_transformer' ...>
```

**Options to resolve:**

- Option A (recommended): Pin the scikit-learn version used for training and use the same version in deployment. Find your local version with:

```powershell
pip show scikit-learn
```

Then pin it in `requirements.txt`, e.g.: `scikit-learn==1.3.2`, and reinstall on the server.

- Option B: Regenerate artifacts on the deployment host by running `python -m pipeline` there (requires the host to have the training dependencies and enough resources).

- Option C: Include prebuilt artifacts with the app (use `scripts/package_artifacts.py` to prepare `deploy_artifacts/`), and deploy those files with the app.

- Fallback: The app attempts to rebuild the preprocessor at runtime if unpickling fails. This can allow predictions to continue but may produce slightly different preprocessing results unless the transformer is fitted on the same data.

---

## Useful commands
- Package artifacts (zip): `python scripts/package_artifacts.py --zip`
- Run full pipeline locally: `python -m pipeline`
- Run test prediction: `python -m test_prediction`
- Run Streamlit UI: `streamlit run app.py`

---

If you'd like, I can also add a `scripts/README.md` with only the packaging/deploy steps or help you pin `scikit-learn` to match your training environment. Which would you prefer next?