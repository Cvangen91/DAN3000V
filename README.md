# DAN3000V

Spotify Data Analytics Project

Overview
This project analyzes a Spotify dataset to understand what factors influence song popularity and to build models that can predict whether a song becomes a “hit”.

Project Structure
spotify-project/
├─ data_raw/                  # Original dataset from Kaggle
├─ data_processed/            # Cleaned dataset (from Tableau Prep)
│   └─ spotify_clean.csv
├─ scripts/
│   └─ Analysis.R             # Main analysis script
├─ outputs/                   # Optional: figures and results
├─ spotify-project.Rproj
└─ README.md

Data Processing
Data cleaning and feature engineering were performed in Tableau Prep.

Steps performed:
- Removed duplicate songs by aggregating on track_name
- Averaged numerical variables (e.g. popularity, energy, tempo)
- Converted duration_ms to duration_min
- Rounded duration values
- Created new variables:
  - is_hit (popularity >= 50)
  - tempo_category (Slow / Medium / Fast)
  - danceability_level (Low / Medium / High)
  - energy_level (Low / Medium / High)

Cleaned dataset location:
data_processed/spotify_clean.csv

Important: Data Format
The dataset uses:
- ; as separator
- , as decimal

Therefore, in R the dataset must be loaded like this:

spotify <- read_delim("data_processed/spotify_clean.csv", delim = ";")

spotify <- spotify %>%
  mutate(across(
    c(acousticness, danceability, energy, instrumentalness,
      liveness, loudness, speechiness, tempo, popularity,
      valence, duration_min),
    ~ as.numeric(gsub(",", ".", .))
  ))

Required Packages
Run this once in R:

install.packages(c(
  "dplyr",
  "readr",
  "ggplot2",
  "tidyr",
  "caret",
  "corrplot",
  "rpart",
  "rpart.plot",
  "broom"
))

Workflow
1. Pull latest changes from GitHub
2. Open the R project (spotify-project.Rproj)
3. Run scripts/Analysis.R

Current Analysis
- Data loading and preprocessing
- Summary statistics and structure checks
- Exploratory analysis:
  - Distribution of popularity
  - Hit vs non-hit songs
  - Relationship between features and popularity
- Correlation matrix

Next Steps
- Logistic regression (predict is_hit)
- Linear regression (predict popularity)
- Decision tree model
- Model evaluation and comparison

Collaboration
- If the dataset is updated:
  - Commit and push the new file in data_processed/
- Other members:
  - Pull changes
  - Run Analysis.R

Notes
- A hit is defined as popularity >= 50 (approximately top 25% of songs)
- This threshold was chosen to avoid class imbalance and improve model performance
