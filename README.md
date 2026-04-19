# 🚀 Spaceship Titanic — Kaggle Classification Challenge

> Predicting which passengers aboard the Spaceship Titanic were transported to an alternate dimension after colliding with a spacetime anomaly, achieving 82.2% accuracy with Random Forest.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square" alt="Matplotlib"/>
  <img src="https://img.shields.io/badge/Seaborn-444876?style=flat-square" alt="Seaborn"/>
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle"/>
</p>

---

## Problem

The Spaceship Titanic — an intergalactic passenger liner carrying ~13,000 passengers — collided with a spacetime anomaly near Alpha Centauri, transporting roughly half its passengers to an alternate dimension. Using data recovered from the ship's damaged computer system, the goal is to predict which passengers were transported to help rescue crews locate the missing and help the spaceship company manage liability.

This is a binary classification problem from [Kaggle's Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic).

## Data

- **Source**: Kaggle competition dataset
- **Size**: ~13,000 passengers
- **Features**: PassengerId, HomePlanet (Europa/Earth/Mars), CryoSleep status, Cabin (deck/num/side), Destination, Age, VIP status, amenity spending (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck)
- **Target**: `Transported` (True/False)

## Approach

**Exploratory Data Analysis**
- Analyzed transport rates by HomePlanet — found strong correlation between home planet and transport likelihood
- CryoSleep passengers were disproportionately transported
- Most passengers were headed to TRAPPIST-1e
- Created `Total Spent` feature aggregating all amenity spending
- Solo travelers were disproportionately likely to stay on the ship

**Feature Engineering & Data Cleaning**
- Split `Cabin` into `deck`, `num`, and `side` — decks and sides showed heterogeneous transport rates
- Extracted group size from PassengerId — groups of 4+ behaved differently than solo travelers
- Addressed 24% of records having missing data using smart imputation:
  - CryoSleep passengers: zeroed out amenity spending (they can't use facilities)
  - Group-based imputation: filled HomePlanet, Destination, and deck using travel companions
  - Reduced missing values by 50% through group imputation alone
  - Used linear regression to predict missing Age values (RMSE 13.43, better than median baseline of 14.73)
  - Remaining gaps filled with mode (categorical) or median (numerical)
- Dropped `Name`, `num`, and `Cabin` as non-predictive features

**Models Evaluated (5-fold cross-validation)**

| Model | Test Accuracy |
|---|---|
| Logistic Regression | 79.14% |
| K-Nearest Neighbors (grid-searched) | 77.5% |
| Gradient Boosting (GBM) | 80.7% |
| **Random Forest** | **82.2%** |

## Key Results

| Metric | Detail |
|---|---|
| **Best Model** | Random Forest — 82.2% test accuracy |
| **Evaluation** | 5-fold cross-validation + held-out test set |
| **Key Predictors** | CryoSleep, deck, HomePlanet, total amenity spending, group size |
| **Missing Data Strategy** | Multi-stage imputation (domain logic → group-based → regression → mode/median) |

## How to Run

```bash
# Clone the repo
git clone https://github.com/patilshan/spaceship-titanic-kaggle.git
cd spaceship-titanic-kaggle

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Launch the notebook
jupyter notebook Spaceship_Titanic.ipynb
```

## Project Structure

```
├── Spaceship_Titanic.pdf    # Full analysis report with EDA, cleaning, and modeling
└── README.md
```

## What I Learned

- Smart imputation matters more than model choice — group-based and domain-logic imputation recovered 50% of missing data and outperformed simple median/mode fills
- Feature engineering from composite fields (splitting Cabin into deck/num/side) unlocked significant predictive signal that raw features didn't provide
- Random Forest outperformed other models here due to its ability to handle mixed feature types, missing data resilience, and resistance to overfitting via bootstrap aggregation

## Team

Built in collaboration with **Nishit Patel** and **Ramni Kotra** as part of General Business 656 at UW Madison's Wisconsin School of Business.

---

<p align="center">
  <a href="https://github.com/patilshan">← Back to profile</a>
</p>
