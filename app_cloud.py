#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Streamlit Grundeinstellungen
# ============================================================

st.set_page_config(
    page_title="Sample Decision Tool",
    page_icon="📊",
    layout="centered"
)

st.title("Sample Decision Tool")


# ============================================================
# Einstellungen
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Die Excel-Datei muss im gleichen Ordner wie app.py liegen.
EXCEL_FILE = BASE_DIR / "Book - Kopie.xlsx"

TARGET_COL = "Order_Conversion"

OTHER_COUNTRY_COLS = [
    "Canada",
    "Romania",
    "Israel",
    "Australia",
    "Poland",
    "South_Africa",
    "Brazil",
    "UAE",
    "China",
]

OTHER_ITEM_COLS = [
    "Gun_Tufted",
    "Table_Tufted",
    "Indo_Tibbetan",
]


# ============================================================
# Preprocessing
# ============================================================

def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Bereitet die Trainingsdaten auf.
    Das gleiche Preprocessing wird später auch für die App-Eingabe nachgebaut.
    """

    if TARGET_COL not in df.columns:
        raise ValueError(f"Die Zielspalte '{TARGET_COL}' wurde nicht gefunden.")

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL]).copy()

    # Länder zu Other_Countries zusammenfassen
    existing_country_cols = [col for col in OTHER_COUNTRY_COLS if col in X.columns]

    if existing_country_cols:
        X["Other_Countries"] = X[existing_country_cols].max(axis=1)
        X = X.drop(columns=existing_country_cols)
    else:
        X["Other_Countries"] = 0

    # Items zu Other_Items zusammenfassen
    existing_item_cols = [col for col in OTHER_ITEM_COLS if col in X.columns]

    if existing_item_cols:
        X["Other_Items"] = X[existing_item_cols].max(axis=1)
        X = X.drop(columns=existing_item_cols)
    else:
        X["Other_Items"] = 0

    # Alle Spalten numerisch machen
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Fehlende Werte auffüllen
    X = X.fillna(0)

    # Binäre Spalten sauber auf 0/1 setzen
    for col in X.columns:
        values = set(X[col].dropna().unique())
        if values.issubset({0, 1, 0.0, 1.0}):
            X[col] = X[col].astype(int)

    return X, y


# ============================================================
# Modell direkt in der App trainieren
# ============================================================

@st.cache_resource
def train_model():
    """
    Lädt die Excel-Datei, führt das Preprocessing aus und trainiert das Modell.
    Durch st.cache_resource passiert das nicht bei jedem Klick neu,
    sondern nur beim Start oder wenn sich der Code ändert.
    """

    if not EXCEL_FILE.exists():
        raise FileNotFoundError(
            f"Excel-Datei nicht gefunden: {EXCEL_FILE}. "
            "Die Datei 'Book - Kopie.xlsx' muss im gleichen Ordner wie app.py liegen."
        )

    df = pd.read_excel(EXCEL_FILE)

    X, y = preprocess_dataframe(df)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                solver="lbfgs"
            )),
        ]
    )

    pipeline.fit(X, y)

    feature_columns = list(X.columns)

    return pipeline, feature_columns


try:
    model, feature_columns = train_model()
except Exception as error:
    st.error("Das Modell konnte nicht trainiert werden.")
    st.exception(error)
    st.stop()


# ============================================================
# Eingaben
# ============================================================

country = st.selectbox(
    "Land auswählen",
    [
        "USA",
        "UK",
        "Italy",
        "Belgium",
        "Romania",
        "Australia",
        "India",
        "Canada",
        "Israel",
        "Poland",
        "South_Africa",
        "Brazil",
        "UAE",
        "China",
    ],
)

item = st.selectbox(
    "Produkttyp",
    [
        "Hand_Tufted",
        "Durry",
        "Double_Back",
        "Handwoven",
        "Knotted",
        "Jacquard",
        "Handloom",
        "Gun_Tufted",
        "Indo_Tibbetan",
        "Power_Loom_Jacquard",
        "Table_Tufted",
    ],
)

shape = st.selectbox(
    "Form",
    ["REC", "Round", "Square"],
)

qty = st.number_input(
    "Quantity Required",
    min_value=1,
    value=1,
)

area = st.number_input(
    "AreaFt",
    min_value=1.0,
    value=10.0,
    step=10.0,
)


# ============================================================
# Prediction
# ============================================================

if st.button("Berechnen"):

    # Alle erwarteten Modell-Spalten mit 0 initialisieren
    daten = {col: 0 for col in feature_columns}

    # Numerische Werte setzen
    if "QtyRequired" in daten:
        daten["QtyRequired"] = qty

    if "AreaFt" in daten:
        daten["AreaFt"] = area

    # Länder setzen
    if country in OTHER_COUNTRY_COLS:
        if "Other_Countries" in daten:
            daten["Other_Countries"] = 1
    else:
        if country in daten:
            daten[country] = 1

    # Items setzen
    if item in OTHER_ITEM_COLS:
        if "Other_Items" in daten:
            daten["Other_Items"] = 1
    else:
        if item in daten:
            daten[item] = 1

    # Shape setzen
    # REC ist Baseline, falls es keine eigene Spalte im Modell gibt.
    if shape in daten:
        daten[shape] = 1

    # DataFrame in exakt der Feature-Reihenfolge des Modells
    eingabe = pd.DataFrame([daten])
    eingabe = eingabe[feature_columns]

    wahrscheinlichkeit = model.predict_proba(eingabe)[0][1]

    if wahrscheinlichkeit >= 0.60:
        empfehlung = "✅ Sample senden"
    elif wahrscheinlichkeit >= 0.40:
        empfehlung = "⚠️ Prüfen"
    else:
        empfehlung = "❌ Kein Sample"

    st.subheader("Ergebnis")

    st.metric(
        "Conversion-Wahrscheinlichkeit",
        f"{wahrscheinlichkeit * 100:.1f}%",
    )

    st.success(empfehlung)
