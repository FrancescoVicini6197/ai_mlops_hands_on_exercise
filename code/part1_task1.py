# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import os

os.makedirs("grafici", exist_ok=True) 
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
X = adult.data.features 
y = adult.data.targets 
  
df = pd.concat([X, y], axis=1)

# %%
print(df.shape)
print("\n")
print(df.head())
print("\n")
df.info()
print("\n")
print(df.dtypes)

# %%
#righe duplicate
n_dup = df.duplicated().sum()
print(f"Righe duplicate: {n_dup} ({n_dup / len(df):.2%})")
df = df.drop_duplicates().reset_index(drop=True)

# %%
#valori mancanti
missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)
report = pd.DataFrame({"mancanti": missing, "percentuale": missing_pct})
print("\nValori mancanti per colonna:")
print(report["mancanti"])

#types check num
print(df.dtypes)
num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

print(df[num_cols].describe().T)

#strange values for capital gain and loss --> check later


# %%
#check cat
cat_cols = df.select_dtypes(exclude="number").columns
for c in cat_cols:
    df[c] = df[c].str.strip().str.capitalize()

cat_cols = cat_cols.drop("income").tolist()

# %%
#check single values anomalies
print(df.nunique())

# income
print(df["income"].unique())
df["income"] = df["income"].str.rstrip(".").map({"<=50k": 0, ">50k": 1})

# ? manipulation
print(df["workclass"].unique())
print(df["workclass"].value_counts())

#new duplicates elimination
df = df.replace("?", "Unknown").fillna("Unknown")

for c in ["workclass", "occupation", "native-country"]:
    print(c, (df[c] == "Unknown").sum())

# quante righe hanno almeno un Unknown, e che % del totale sono
rows_unknown = df[(df["workclass"] == "Unknown") | (df["occupation"] == "Unknown") | (df["native-country"] == "Unknown")]
print("% righe:", round(len(rows_unknown) / len(df) * 100, 1))

#print(df["workclass"].value_counts())
n_dup = df.duplicated().sum() + n_dup
print(f"Righe duplicate: {n_dup} ({n_dup / len(df):.2%})")
df = df.drop_duplicates().reset_index(drop=True)

#check education and education num
print(df["education-num"].unique())
print(df["education"].unique())
#one can be dropped later

# %%
# check impossibile values
impossible_values = {
    "age <= 0 o > 100":      ((df["age"] <= 0) | (df["age"] > 120)).sum(),
    "hours <= 0 o > 168":    ((df["hours-per-week"] <= 0) | (df["hours-per-week"] > 168)).sum(),
    "negative capital":      ((df["capital-gain"] < 0) | (df["capital-loss"] < 0)).sum(),
    "both loss and gain":    ((df["capital-gain"] > 0) & (df["capital-loss"] > 0)).sum(),
    "education-num fuori 1-16": ((df["education-num"] < 1) | (df["education-num"] > 16)).sum(),
}

print("Impossibili:", impossible_values)
#so no impossible values

# %%
#target unbalanced?
print(df["income"].value_counts(normalize=True).round(3))
# yes so will need stratification


#%%
num = df.select_dtypes(include="number").drop(columns="income")
num.hist(bins=40, figsize=(14, 8))
plt.suptitle("Single variable distribution"); plt.tight_layout(); 
plt.savefig("grafici/01_numerical_distributions.png", bbox_inches="tight")
plt.show()
print("Skewness:\n", num.skew().round(2))

fig, axes = plt.subplots(3, 3, figsize=(16, 13))
for ax, c in zip(axes.flat, cat_cols):
    df[c].value_counts().head(12).plot.bar(ax=ax, title=c)
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
plt.tight_layout(); 
plt.savefig("grafici/02_categorical_counts.png", bbox_inches="tight");
plt.show()


#%%
print(df.groupby("income")[num.columns].mean().T.round(1))
 
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, c in zip(axes.flat, num.columns):
    sns.boxplot(data=df, x="income", y=c, ax=ax, showfliers=False)
plt.tight_layout(); 
plt.savefig("grafici/03_numerical_vs_income.png", bbox_inches="tight");
plt.show()

#%%

fig, axes = plt.subplots(3, 3, figsize=(16, 13))
for ax, c in zip(axes.flat, cat_cols):
    perc_alto = df.groupby(c)["income"].mean() * 100
    perc_alto.sort_values(ascending=False).plot.bar(ax=ax, title=c)
    ax.set_ylabel("% >50K")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout(); 
plt.savefig("grafici/04_categorical_vs_income.png", bbox_inches="tight");
plt.show()


corr = df[num.columns.tolist() + ["income"]].corr(method="spearman")
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlazione tra variabili numeriche"); 
plt.savefig("grafici/05_correlations.png", bbox_inches="tight");
plt.show()


#%%
# Regression model:
#1. capital-gain and capital-loss are too big, I use log to reduce them
df["capital-gain"] = np.log1p(df["capital-gain"])
df["capital-loss"] = np.log1p(df["capital-loss"])


#    remove fnlwgt, education and income for data leakage
X = df.drop(columns=["income", "fnlwgt", "education"])
X = pd.get_dummies(X, drop_first=True, dtype=int)   # testo -> colonne 0/1
y = df["income"]

# 3. 80% train, 20% test (stratify =y because unbalanced target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. train model
modello = LogisticRegression(max_iter=5000, class_weight="balanced")
modello.fit(X_train, y_train)

# 5. test
y_pred = modello.predict(X_test)
print(classification_report(y_test, y_pred))
print("AUC:", round(roc_auc_score(y_test, modello.predict_proba(X_test)[:, 1]), 3))