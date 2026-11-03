# CINEMETRICS — Movie Analytics Dashboard

A redesigned interactive Streamlit dashboard for the Movie Success Predictor project.

## Included
- `app.py`: redesigned dashboard
- `data/`: original, cleaned, engineered, encoded, scaled, selected-feature and PCA CSV files
- `notebooks/`: supplied preprocessing and analysis notebooks
- `requirements.txt`: dependencies

## Run in VS Code (Windows)

Open this folder in VS Code and run:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the Local URL printed in the terminal (normally `http://localhost:8501`). If asked for an email, leave it blank and press Enter.

## Dashboard
- **Discover:** global year/country/rating/genre filters, searchable catalogue, KPIs and overview charts.
- **Explore:** budget, gross, runtime, genre, country, rating and gross-to-budget comparisons.
- **Preparation lab:** cleaning, feature engineering, encoding and scaling in grouped tabs.
- **Model inputs:** feature selection, PCA exploration and training hand-off.

The sidebar filters apply to the original movie data used in the discovery and exploration views. Transformed CSVs are shown as the saved outputs from your notebooks; they do not all contain a stable row identifier for safely joining every global filter to each transformed matrix.
