from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# Symptom and disease lists kept from original project.
l1 = [
    "back_pain",
    "constipation",
    "abdominal_pain",
    "diarrhoea",
    "mild_fever",
    "yellow_urine",
    "yellowing_of_eyes",
    "acute_liver_failure",
    "fluid_overload",
    "swelling_of_stomach",
    "swelled_lymph_nodes",
    "malaise",
    "blurred_and_distorted_vision",
    "phlegm",
    "throat_irritation",
    "redness_of_eyes",
    "sinus_pressure",
    "runny_nose",
    "congestion",
    "chest_pain",
    "weakness_in_limbs",
    "fast_heart_rate",
    "pain_during_bowel_movements",
    "pain_in_anal_region",
    "bloody_stool",
    "irritation_in_anus",
    "neck_pain",
    "dizziness",
    "cramps",
    "bruising",
    "obesity",
    "swollen_legs",
    "swollen_blood_vessels",
    "puffy_face_and_eyes",
    "enlarged_thyroid",
    "brittle_nails",
    "swollen_extremeties",
    "excessive_hunger",
    "extra_marital_contacts",
    "drying_and_tingling_lips",
    "slurred_speech",
    "knee_pain",
    "hip_joint_pain",
    "muscle_weakness",
    "stiff_neck",
    "swelling_joints",
    "movement_stiffness",
    "spinning_movements",
    "loss_of_balance",
    "unsteadiness",
    "weakness_of_one_body_side",
    "loss_of_smell",
    "bladder_discomfort",
    "foul_smell_of urine",
    "continuous_feel_of_urine",
    "passage_of_gases",
    "internal_itching",
    "toxic_look_(typhos)",
    "depression",
    "irritability",
    "muscle_pain",
    "altered_sensorium",
    "red_spots_over_body",
    "belly_pain",
    "abnormal_menstruation",
    "dischromic _patches",
    "watering_from_eyes",
    "increased_appetite",
    "polyuria",
    "family_history",
    "mucoid_sputum",
    "rusty_sputum",
    "lack_of_concentration",
    "visual_disturbances",
    "receiving_blood_transfusion",
    "receiving_unsterile_injections",
    "coma",
    "stomach_bleeding",
    "distention_of_abdomen",
    "history_of_alcohol_consumption",
    "fluid_overload",
    "blood_in_sputum",
    "prominent_veins_on_calf",
    "palpitations",
    "painful_walking",
    "pus_filled_pimples",
    "blackheads",
    "scurring",
    "skin_peeling",
    "silver_like_dusting",
    "small_dents_in_nails",
    "inflammatory_nails",
    "blister",
    "red_sore_around_nose",
    "yellow_crust_ooze",
]

disease = [
    "Fungal infection",
    "Allergy",
    "GERD",
    "Chronic cholestasis",
    "Drug Reaction",
    "Peptic ulcer disease",
    "AIDS",
    "Diabetes",
    "Gastroenteritis",
    "Bronchial Asthma",
    "Hypertension",
    "Migraine",
    "Cervical spondylosis",
    "Paralysis (brain hemorrhage)",
    "Jaundice",
    "Malaria",
    "Chicken pox",
    "Dengue",
    "Typhoid",
    "Hepatitis A",
    "Hepatitis B",
    "Hepatitis C",
    "Hepatitis D",
    "Hepatitis E",
    "Alcoholic hepatitis",
    "Tuberculosis",
    "Common Cold",
    "Pneumonia",
    "Dimorphic hemorrhoids (piles)",
    "Heart attack",
    "Varicose veins",
    "Hypothyroidism",
    "Hyperthyroidism",
    "Hypoglycemia",
    "Osteoarthritis",
    "Arthritis",
    "(Vertigo) Paroxysmal Positional Vertigo",
    "Acne",
    "Urinary tract infection",
    "Psoriasis",
    "Impetigo",
]

PROGNOSIS_MAP = {
    "Fungal infection": 0,
    "Allergy": 1,
    "GERD": 2,
    "Chronic cholestasis": 3,
    "Drug Reaction": 4,
    "Peptic ulcer diseae": 5,
    "AIDS": 6,
    "Diabetes ": 7,
    "Gastroenteritis": 8,
    "Bronchial Asthma": 9,
    "Hypertension ": 10,
    "Migraine": 11,
    "Cervical spondylosis": 12,
    "Paralysis (brain hemorrhage)": 13,
    "Jaundice": 14,
    "Malaria": 15,
    "Chicken pox": 16,
    "Dengue": 17,
    "Typhoid": 18,
    "hepatitis A": 19,
    "Hepatitis B": 20,
    "Hepatitis C": 21,
    "Hepatitis D": 22,
    "Hepatitis E": 23,
    "Alcoholic hepatitis": 24,
    "Tuberculosis": 25,
    "Common Cold": 26,
    "Pneumonia": 27,
    "Dimorphic hemmorhoids(piles)": 28,
    "Heart attack": 29,
    "Varicose veins": 30,
    "Hypothyroidism": 31,
    "Hyperthyroidism": 32,
    "Hypoglycemia": 33,
    "Osteoarthristis": 34,
    "Arthritis": 35,
    "(vertigo) Paroymsal  Positional Vertigo": 36,
    "Acne": 37,
    "Urinary tract infection": 38,
    "Psoriasis": 39,
    "Impetigo": 40,
}

ALGO_COLORS = {
    "Decision Tree": "#58A6FF",
    "Random Forest": "#3FB950",
    "Naive Bayes": "#BC8CFF",
}

BASE_DIR = Path(__file__).resolve().parent


# Resolve CSV files relative to this script so it works from any working directory.
def _load_data():
    train_path = BASE_DIR / "Training.csv"
    test_path = BASE_DIR / "Testing.csv"

    df = pd.read_csv(train_path)
    df["prognosis"] = df["prognosis"].replace(PROGNOSIS_MAP).infer_objects(copy=False)
    df = df.dropna(subset=["prognosis"])  # Drop rows whose prognosis did not map.

    tr = pd.read_csv(test_path)
    tr["prognosis"] = tr["prognosis"].replace(PROGNOSIS_MAP).infer_objects(copy=False)
    tr = tr.dropna(subset=["prognosis"])

    X_train = df[l1]
    y_train = df["prognosis"].astype(int)
    X_test = tr[l1]
    y_test = tr["prognosis"].astype(int)
    return X_train, y_train, X_test, y_test


def _train_models(X_train, y_train):
    models = {
        "Decision Tree": tree.DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
        "Naive Bayes": GaussianNB(),
    }
    for clf in models.values():
        clf.fit(X_train, y_train)
    return models


def _compute_accuracy(models, X_test, y_test):
    return {
        algo: float(accuracy_score(y_test, clf.predict(X_test)))
        for algo, clf in models.items()
    }


def _symptom_vector(selected_symptoms):
    vec = [0] * len(l1)
    for idx, sym in enumerate(l1):
        if sym in selected_symptoms:
            vec[idx] = 1
    return np.array([vec])


def _predict_disease(models, algo, selected_symptoms):
    inp = _symptom_vector(selected_symptoms)
    pred = int(models[algo].predict(inp)[0])
    if 0 <= pred < len(disease):
        return disease[pred]
    return "Not Found"


def _predict_live_metrics(models, algo, selected_symptoms):
    clf = models[algo]
    inp = _symptom_vector(selected_symptoms)

    started = perf_counter()
    pred = int(clf.predict(inp)[0])
    latency_ms = (perf_counter() - started) * 1000.0

    confidence = 0.0
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(inp)[0]
        confidence = float(np.max(proba))

    disease_name = disease[pred] if 0 <= pred < len(disease) else "Not Found"
    return {
        "disease": disease_name,
        "accuracy": MODEL_ACCURACY[algo],
        "confidence": confidence,
        "latency_ms": latency_ms,
    }


X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = _load_data()
MODELS = _train_models(X_TRAIN, Y_TRAIN)
MODEL_ACCURACY = _compute_accuracy(MODELS, X_TEST, Y_TEST)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    selected_values = ["-- Select Symptom --"] * 5
    patient_name = ""
    status = "Awaiting prediction..."
    status_type = "info"
    results = {}

    if request.method == "POST":
        patient_name = request.form.get("patient_name", "").strip()
        selected_values = [
            request.form.get(f"symptom{i}", "-- Select Symptom --") for i in range(1, 6)
        ]
        selected = [s for s in selected_values if s != "-- Select Symptom --"]
        selected = list(dict.fromkeys(selected))

        action = request.form.get("action", "")
        if not selected:
            status = "Please select at least one symptom."
            status_type = "warning"
        elif action in ALGO_COLORS:
            results[action] = _predict_live_metrics(MODELS, action, selected)
            name = patient_name or "Patient"
            status = f"Prediction complete for {name}"
            status_type = "success"
        elif action == "Run All Algorithms":
            for algo in ALGO_COLORS:
                results[algo] = _predict_live_metrics(MODELS, algo, selected)
            name = patient_name or "Patient"
            status = f"All algorithms complete for {name}"
            status_type = "success"
        else:
            status = "Select symptoms and choose an algorithm to continue."
            status_type = "warning"

    return render_template(
        "index.html",
        symptom_options=sorted(l1),
        selected_values=selected_values,
        patient_name=patient_name,
        status=status,
        status_type=status_type,
        results=results,
        algo_colors=ALGO_COLORS,
        model_accuracy=MODEL_ACCURACY,
    )


if __name__ == "__main__":
    app.run(debug=False)
