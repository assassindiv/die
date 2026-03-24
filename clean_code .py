import tkinter as tk
from tkinter import ttk, font
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import threading

# ─────────────────────────────────────────────
#  SYMPTOM & DISEASE LISTS (unchanged from original)
# ─────────────────────────────────────────────
l1 = [
    'back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite',
    'polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration',
    'visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma',
    'stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload',
    'blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking',
    'pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting',
    'small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze'
]

disease = [
    'Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
    'Peptic ulcer disease','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
    'Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)','Jaundice','Malaria',
    'Chicken pox','Dengue','Typhoid','Hepatitis A','Hepatitis B','Hepatitis C','Hepatitis D',
    'Hepatitis E','Alcoholic hepatitis','Tuberculosis','Common Cold','Pneumonia',
    'Dimorphic hemorrhoids (piles)','Heart attack','Varicose veins','Hypothyroidism',
    'Hyperthyroidism','Hypoglycemia','Osteoarthritis','Arthritis',
    '(Vertigo) Paroxysmal Positional Vertigo','Acne','Urinary tract infection','Psoriasis','Impetigo'
]

PROGNOSIS_MAP = {
    'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,
    'Hypertension ':10,'Migraine':11,'Cervical spondylosis':12,'Paralysis (brain hemorrhage)':13,
    'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,
    'Tuberculosis':25,'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,
    'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,'Hyperthyroidism':32,
    'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,
    'Psoriasis':39,'Impetigo':40
}

l2 = [0] * len(l1)

# ─────────────────────────────────────────────
#  LOAD & PREPARE DATA
# ─────────────────────────────────────────────
df = pd.read_csv("Training.csv")
df.replace({'prognosis': PROGNOSIS_MAP}, inplace=True)
X = df[l1]
y = df[["prognosis"]]

tr = pd.read_csv("Testing.csv")
tr.replace({'prognosis': PROGNOSIS_MAP}, inplace=True)
X_test = tr[l1]
y_test = tr[["prognosis"]]

# ─────────────────────────────────────────────
#  COLOUR PALETTE & CONSTANTS
# ─────────────────────────────────────────────
BG        = "#0D1117"
CARD      = "#161B22"
CARD2     = "#1C2333"
ACCENT    = "#58A6FF"
ACCENT2   = "#3FB950"
WARN      = "#F0883E"
TEXT      = "#E6EDF3"
MUTED     = "#8B949E"
BORDER    = "#30363D"
RED       = "#FF6B6B"
TEAL      = "#39D0D8"
PURPLE    = "#BC8CFF"

ALGO_COLORS = {
    "Decision Tree":  ACCENT,
    "Random Forest":  ACCENT2,
    "Naive Bayes":    PURPLE,
}

# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
class DiseasePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MedPredict — Disease Prediction System")
        self.root.configure(bg=BG)
        self.root.geometry("960x780")
        self.root.minsize(860, 700)

        self.symptom_vars = [tk.StringVar(value="-- Select Symptom --") for _ in range(5)]
        self.results      = {}          # algo → (disease_str, accuracy_float)
        self.is_loading   = False

        self._setup_styles()
        self._build_ui()

    # ── Styles ───────────────────────────────
    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use("clam")

        s.configure("TCombobox",
                    fieldbackground=CARD2, background=CARD2,
                    foreground=TEXT, selectbackground=CARD2,
                    selectforeground=TEXT, bordercolor=BORDER,
                    arrowcolor=ACCENT, font=("Segoe UI", 10))
        s.map("TCombobox",
              fieldbackground=[("readonly", CARD2)],
              foreground=[("readonly", TEXT)],
              bordercolor=[("focus", ACCENT)])

        s.configure("Vertical.TScrollbar", background=CARD, troughcolor=BG,
                    bordercolor=BORDER, arrowcolor=MUTED)

    # ── Build full UI ─────────────────────────
    def _build_ui(self):
        # ─ outer padding frame
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=24, pady=20)

        self._build_header(outer)
        self._build_body(outer)
        self._build_footer(outer)

    # ── Header ───────────────────────────────
    def _build_header(self, parent):
        hdr = tk.Frame(parent, bg=BG)
        hdr.pack(fill="x", pady=(0, 20))

        # Pill badge
        badge = tk.Frame(hdr, bg="#1F3A5F", bd=0)
        badge.pack(anchor="w")
        tk.Label(badge, text="  ● ML-POWERED DIAGNOSIS  ", bg="#1F3A5F",
                 fg=ACCENT, font=("Segoe UI", 8, "bold"), padx=6, pady=3).pack()

        tk.Label(hdr, text="MedPredict",
                 bg=BG, fg=TEXT,
                 font=("Georgia", 34, "bold")).pack(anchor="w", pady=(6,0))

        tk.Label(hdr, text="Intelligent Disease Prediction using 3 ML Algorithms",
                 bg=BG, fg=MUTED,
                 font=("Segoe UI", 11)).pack(anchor="w")

        # divider
        tk.Frame(hdr, bg=BORDER, height=1).pack(fill="x", pady=(14, 0))

    # ── Body (two-column) ────────────────────
    def _build_body(self, parent):
        body = tk.Frame(parent, bg=BG)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        self._build_input_card(body)
        self._build_result_panel(body)

    # ── Input Card (left) ────────────────────
    def _build_input_card(self, parent):
        card = tk.Frame(parent, bg=CARD, bd=0, relief="flat")
        card.grid(row=0, column=0, sticky="nsew", padx=(0,12), pady=0)

        # card header
        ch = tk.Frame(card, bg=CARD)
        ch.pack(fill="x", padx=20, pady=(18,0))
        tk.Label(ch, text="Patient Details", bg=CARD, fg=TEXT,
                 font=("Segoe UI", 13, "bold")).pack(side="left")
        tk.Label(ch, text="Select up to 5 symptoms", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="right", pady=(4,0))

        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(10,0))

        form = tk.Frame(card, bg=CARD)
        form.pack(fill="x", padx=20, pady=16)

        # Patient name
        self._field_label(form, "Patient Name", row=0)
        self.name_var = tk.StringVar()
        name_entry = tk.Entry(form, textvariable=self.name_var,
                              bg=CARD2, fg=TEXT, insertbackground=ACCENT,
                              relief="flat", font=("Segoe UI", 10),
                              highlightthickness=1, highlightcolor=ACCENT,
                              highlightbackground=BORDER)
        name_entry.grid(row=1, column=0, sticky="ew", pady=(4,14), ipady=6)
        form.columnconfigure(0, weight=1)

        # Symptoms
        OPTIONS = ["-- Select Symptom --"] + sorted(l1)
        for i, var in enumerate(self.symptom_vars):
            self._field_label(form, f"Symptom {i+1}", row=(i+1)*2)
            cb = ttk.Combobox(form, textvariable=var, values=OPTIONS,
                              state="readonly", font=("Segoe UI", 10))
            cb.grid(row=(i+1)*2+1, column=0, sticky="ew", pady=(4,12), ipady=4)

        # divider
        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=20)

        # Buttons
        btn_row = tk.Frame(card, bg=CARD)
        btn_row.pack(fill="x", padx=20, pady=16)

        for algo, color in ALGO_COLORS.items():
            self._algo_btn(btn_row, algo, color)

        # big predict-all button
        all_btn = tk.Frame(card, bg=CARD)
        all_btn.pack(fill="x", padx=20, pady=(0,20))
        self._make_button(all_btn, "⚡  Run All Algorithms", self._run_all, WARN,
                          "#0D1117", full=True)

    def _field_label(self, parent, text, row):
        tk.Label(parent, text=text.upper(), bg=CARD, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).grid(row=row, column=0,
                                                     sticky="w", pady=(0,0))

    def _algo_btn(self, parent, algo, color):
        short = algo.split()[0]          # "Decision", "Random", "Naive"
        cmd = lambda a=algo: self._run_single(a)
        btn = tk.Button(parent, text=algo,
                        bg=CARD2, fg=color,
                        activebackground=BORDER, activeforeground=color,
                        relief="flat", font=("Segoe UI", 9, "bold"),
                        cursor="hand2", pady=7, padx=10,
                        highlightthickness=1, highlightbackground=color,
                        command=cmd)
        btn.pack(side="left", fill="x", expand=True, padx=(0,8))

    def _make_button(self, parent, text, cmd, bg, fg, full=False):
        btn = tk.Button(parent, text=text, command=cmd,
                        bg=bg, fg=fg, activebackground=ACCENT,
                        relief="flat", font=("Segoe UI", 10, "bold"),
                        cursor="hand2", pady=9,
                        highlightthickness=0)
        if full:
            btn.pack(fill="x")
        else:
            btn.pack(side="left", padx=(0,8))
        return btn

    # ── Result Panel (right) ─────────────────
    def _build_result_panel(self, parent):
        panel = tk.Frame(parent, bg=BG)
        panel.grid(row=0, column=1, sticky="nsew")

        # Status label
        self.status_var = tk.StringVar(value="Awaiting prediction…")
        tk.Label(panel, textvariable=self.status_var,
                 bg=BG, fg=MUTED, font=("Segoe UI", 9, "italic")).pack(anchor="w", pady=(0,10))

        self.result_cards = {}
        for algo, color in ALGO_COLORS.items():
            self.result_cards[algo] = self._result_card(panel, algo, color)

        # Accuracy summary card
        self._build_accuracy_card(panel)

    def _result_card(self, parent, algo, color):
        frame = tk.Frame(parent, bg=CARD, bd=0)
        frame.pack(fill="x", pady=(0, 10))

        top = tk.Frame(frame, bg=color, height=3)
        top.pack(fill="x")

        inner = tk.Frame(frame, bg=CARD)
        inner.pack(fill="x", padx=14, pady=12)

        tk.Label(inner, text=algo.upper(), bg=CARD, fg=color,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")

        result_var = tk.StringVar(value="—")
        lbl = tk.Label(inner, textvariable=result_var,
                       bg=CARD, fg=TEXT,
                       font=("Segoe UI", 15, "bold"),
                       wraplength=240, justify="left")
        lbl.pack(anchor="w", pady=(4,2))

        acc_var = tk.StringVar(value="")
        tk.Label(inner, textvariable=acc_var, bg=CARD, fg=MUTED,
                 font=("Segoe UI", 8)).pack(anchor="w")

        return {"result": result_var, "accuracy": acc_var, "frame": frame, "color": color}

    def _build_accuracy_card(self, parent):
        self.acc_card = tk.Frame(parent, bg=CARD)
        self.acc_card.pack(fill="x", pady=(4,0))

        tk.Frame(self.acc_card, bg=BORDER, height=1).pack(fill="x")
        inner = tk.Frame(self.acc_card, bg=CARD)
        inner.pack(fill="x", padx=14, pady=12)

        tk.Label(inner, text="MODEL ACCURACY", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0,8))

        self.acc_bars = {}
        for algo, color in ALGO_COLORS.items():
            row = tk.Frame(inner, bg=CARD)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=algo[:12], bg=CARD, fg=TEXT,
                     font=("Segoe UI", 8), width=13, anchor="w").pack(side="left")
            bar_bg = tk.Frame(row, bg=BORDER, height=8)
            bar_bg.pack(side="left", fill="x", expand=True, padx=(4,6))
            bar_fg = tk.Frame(bar_bg, bg=BORDER, height=8, width=0)
            bar_fg.place(x=0, y=0, relheight=1)
            pct_lbl = tk.Label(row, text="—", bg=CARD, fg=color,
                               font=("Segoe UI", 8, "bold"), width=6)
            pct_lbl.pack(side="left")
            self.acc_bars[algo] = (bar_bg, bar_fg, pct_lbl, color)

    # ── Footer ───────────────────────────────
    def _build_footer(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=(14,6))
        ft = tk.Frame(parent, bg=BG)
        ft.pack(fill="x")
        tk.Label(ft, text="⚠  This tool is for educational purposes only. Always consult a qualified physician.",
                 bg=BG, fg=MUTED, font=("Segoe UI", 8)).pack(side="left")
        tk.Label(ft, text="41 Diseases · 3 Algorithms",
                 bg=BG, fg=BORDER, font=("Segoe UI", 8)).pack(side="right")

    # ─────────────────────────────────────────
    #  ML PREDICTION LOGIC (original, unchanged)
    # ─────────────────────────────────────────
    def _get_symptom_vector(self):
        vec = [0] * len(l1)
        selected = [v.get() for v in self.symptom_vars
                    if v.get() != "-- Select Symptom --"]
        for k, sym in enumerate(l1):
            if sym in selected:
                vec[k] = 1
        return [vec]

    def _predict(self, algo):
        inp = self._get_symptom_vector()
        if algo == "Decision Tree":
            clf = tree.DecisionTreeClassifier()
            clf.fit(X, y)
            pred   = clf.predict(inp)[0]
            y_pred = clf.predict(X_test)
        elif algo == "Random Forest":
            clf = RandomForestClassifier()
            clf.fit(X, np.ravel(y))
            pred   = clf.predict(inp)[0]
            y_pred = clf.predict(X_test)
        else:  # Naive Bayes
            clf = GaussianNB()
            clf.fit(X, np.ravel(y))
            pred   = clf.predict(inp)[0]
            y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        dis = disease[pred] if 0 <= pred < len(disease) else "Not Found"
        return dis, acc

    # ─────────────────────────────────────────
    #  BUTTON HANDLERS
    # ─────────────────────────────────────────
    def _run_single(self, algo):
        if self.is_loading:
            return
        self.is_loading = True
        self.status_var.set(f"Running {algo}…")
        threading.Thread(target=self._worker_single, args=(algo,), daemon=True).start()

    def _worker_single(self, algo):
        try:
            dis, acc = self._predict(algo)
            self.root.after(0, self._update_card, algo, dis, acc)
        except Exception as e:
            self.root.after(0, self.status_var.set, f"Error: {e}")
        finally:
            self.is_loading = False

    def _run_all(self):
        if self.is_loading:
            return
        self.is_loading = True
        self.status_var.set("Running all algorithms…")
        threading.Thread(target=self._worker_all, daemon=True).start()

    def _worker_all(self):
        results = {}
        try:
            for algo in ALGO_COLORS:
                dis, acc = self._predict(algo)
                results[algo] = (dis, acc)
            self.root.after(0, self._update_all, results)
        except Exception as e:
            self.root.after(0, self.status_var.set, f"Error: {e}")
        finally:
            self.is_loading = False

    # ─────────────────────────────────────────
    #  UI UPDATE HELPERS
    # ─────────────────────────────────────────
    def _update_card(self, algo, dis, acc):
        c = self.result_cards[algo]
        c["result"].set(dis)
        c["accuracy"].set(f"Test accuracy: {acc*100:.1f}%")
        self._animate_bar(algo, acc)
        name = self.name_var.get().strip() or "Patient"
        self.status_var.set(f"Prediction complete for {name}")

    def _update_all(self, results):
        for algo, (dis, acc) in results.items():
            c = self.result_cards[algo]
            c["result"].set(dis)
            c["accuracy"].set(f"Test accuracy: {acc*100:.1f}%")
            self._animate_bar(algo, acc)
        name = self.name_var.get().strip() or "Patient"
        self.status_var.set(f"All algorithms complete for {name}")

    def _animate_bar(self, algo, acc, step=0):
        bar_bg, bar_fg, pct_lbl, color = self.acc_bars[algo]
        target_pct = acc * 100
        cur_pct = step / 20 * target_pct  # animate over 20 steps
        bar_bg.update_idletasks()
        w = bar_bg.winfo_width()
        bar_fg.config(bg=color, width=max(1, int(w * cur_pct / 100)))
        pct_lbl.config(text=f"{cur_pct:.0f}%")
        if step < 20:
            self.root.after(18, self._animate_bar, algo, acc, step + 1)
        else:
            pct_lbl.config(text=f"{target_pct:.1f}%")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = DiseasePredictorApp(root)
    root.mainloop()
