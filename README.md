# 🐧 Penguin Species Classification: Decision Tree vs KNN

Mini-project 1b of my AI/ML Engineer training (MSIT, Advanced 1: Machine Learning).
**Goal:** predict a penguin's species (**Adelie, Chinstrap or Gentoo**) from its body measurements, island and sex.

📓 **View the notebook:** [nbviewer](https://nbviewer.org/github/esosaphilip/Penguin_species/blob/main/penguin_species.ipynb) · [Open in Colab](https://colab.research.google.com/github/esosaphilip/Penguin_species/blob/main/penguin_species.ipynb)

**Type of problem:** multi-class classification (3 classes, supervised learning)
**Special rule for this project:** built from a **blank notebook, without AI assistants**, as a test of what I can do on my own (only the scikit-learn / pandas documentation and my own notes were allowed).

---

## 📊 Results at a glance

| Model | Train accuracy | Test accuracy | 5-fold CV accuracy |
|---|---|---|---|
| Decision Tree (`max_depth=6`) | 1.000 | 0.986 | 0.956 ± 0.015 |
| **KNN** (k = 5) | 0.996 | 0.986 | **0.985 ± 0.014** |

**Main finding:** both models classify **68 of 69** test penguins correctly. Cross-validation breaks the tie: **KNN is about 3 points better** (0.985 vs 0.956) and the decision tree overfits slightly (train 100%).

---

## 🗂️ Data

- **Source:** the Palmer Penguins dataset that ships with seaborn (`sns.load_dataset('penguins')`)
- **Size:** 344 penguins × 7 columns
- **Target (`y`):** `species`: Adelie 152, Gentoo 124, Chinstrap 68 (mildly imbalanced → stratified split)
- **Features (`X`):** `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex`
- **Missing values:** 2 rows with no measurements at all; `sex` missing in 11 rows

---

## ⚙️ Method

1. **Load & explore:** shape, data types, missing values, penguins per species
2. **X and y:** species is the target, so it is **not** in X
3. **Train-test split:** 80 / 20, `random_state=42`, **`stratify=y`** (keeps the species ratio in both sets)
4. **Preprocessing in a `Pipeline` + `ColumnTransformer`** (fitted on training data only → no data leakage):

   | Column group | Steps |
   |---|---|
   | numeric: bill length, bill depth, flipper length, body mass | `SimpleImputer(mean)` → `StandardScaler` |
   | categorical: `island`, `sex` | `SimpleImputer(most_frequent)` → `OneHotEncoder` |

5. **Two models**, each in its own pipeline (with `clone(preprocessor)`): `DecisionTreeClassifier` and `KNeighborsClassifier`
6. **Evaluation:** train vs test accuracy, 3×3 confusion matrices, 5-fold cross-validation

**Why scaling matters here:** KNN measures distances. Without scaling, `body_mass_g` (thousands) would drown out `bill_depth_mm` (tens).

---

## 🔍 The two mistakes

| Model | Penguin | Real | Predicted | Why it's hard |
|---|---|---|---|---|
| Decision Tree | Torgersen, bill 44.1 mm, **flipper 210 mm**, 4,000 g | Adelie | Gentoo | an unusually large Adelie: its flipper length is closer to the Gentoo average (217 mm) than to the Adelie average (190 mm) |
| KNN | Dream, **bill 42.2 mm**, flipper 180 mm | Adelie | Chinstrap | a long bill for an Adelie, on the island where the Chinstraps live, so its nearest neighbours were Chinstraps |

Both models fail on **unusual edge cases**, not on typical penguins.

---

## 💡 What I learned

- **Read the task first.** I first chose the wrong target column; a perfect model for the wrong question is worthless.
- **Pipeline vs ColumnTransformer:** a `Pipeline` runs steps **one after another** (impute → scale → model); a `ColumnTransformer` runs transformers **side by side** on different column groups.
- **Column names, not data:** a `ColumnTransformer` needs lists of column **names**, not DataFrames.
- **Reading error messages** (`too many values to unpack (expected 2)`, `A given column is not a column of the dataframe`) points straight to the bug.
- **Penguins are easier than the Titanic** because the species really differ in body shape (Gentoo is much heavier, with longer flippers). Titanic survival also depended on chance.
- **Equal test scores ≠ equal models.** Cross-validation showed KNN generalises better.

---

## ▶️ How to run

1. Open `ML__penguin_species.ipynb` in Google Colab or Jupyter
2. `pip install -r requirements.txt`
3. **Runtime → Restart session and run all**

**Tools:** Python · pandas · seaborn · scikit-learn · matplotlib

---

*Author: Eseosa Philip Edosomwan, Wuppertal, Germany · AI/ML Engineer in training*
