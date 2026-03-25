# Disease Prediction Using Flask

This project predicts a probable disease from selected symptoms using three machine learning algorithms:

- Decision Tree
- Random Forest
- Naive Bayes

The user interface is now a Flask web app, so it runs in your browser instead of a Tkinter desktop window.

## Files

- `clean_code .py` - Flask app and ML prediction logic
- `templates/index.html` - Web UI template
- `Training.csv` - Training dataset
- `Testing.csv` - Testing dataset

## Run Locally

1. Open terminal in this folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
python "clean_code .py"
```

4. Open this URL:

```text
http://127.0.0.1:5000
```

## How To Use

1. Enter patient name (optional).
2. Select up to 5 symptoms.
3. Click one algorithm button or `Run All Algorithms`.
4. View predicted disease and model accuracy cards on the right.


