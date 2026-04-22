# Analysis.R
# Spotify Data Analytics Project
# This script loads the cleaned Spotify dataset, prepares the data,
# performs exploratory analysis, and creates initial visualizations.

# -----------------------------
# 1. Load required packages
# -----------------------------
library(dplyr)
library(readr)
library(ggplot2)
library(tidyr)
library(caret)
library(corrplot)
library(rpart)
library(rpart.plot)
library(broom)

# -----------------------------
# 2. Load dataset
# -----------------------------
# Note:
# The dataset uses:
# - ";" as separator
# - "," as decimal
spotify <- read_delim("data_processed/spotify_clean.csv", delim = ";")

# -----------------------------
# 3. Convert numeric columns
# -----------------------------
# Some numeric variables are imported as text because of decimal commas.
spotify <- spotify %>%
  mutate(across(
    c(acousticness, danceability, energy, instrumentalness,
      liveness, loudness, speechiness, tempo, popularity,
      valence, duration_min),
    ~ as.numeric(gsub(",", ".", .))
  ))

# -----------------------------
# 4. Check data structure
# -----------------------------
glimpse(spotify)
summary(spotify)
head(spotify)

# -----------------------------
# 5. Prepare variables
# -----------------------------
# Convert is_hit to factor for classification analysis
spotify$is_hit <- as.factor(spotify$is_hit)

# Optional: check class balance
table(spotify$is_hit)
prop.table(table(spotify$is_hit))

# -----------------------------
# 6. Exploratory Data Analysis
# -----------------------------

# 6.1 Distribution of popularity
ggplot(spotify, aes(x = popularity)) +
  geom_histogram(bins = 30) +
  theme_minimal() +
  labs(
    title = "Distribution of Song Popularity",
    x = "Popularity",
    y = "Count"
  )

# 6.2 Distribution of hit vs not hit
ggplot(spotify, aes(x = is_hit)) +
  geom_bar() +
  theme_minimal() +
  labs(
    title = "Hit vs Not Hit Songs",
    x = "Hit Status",
    y = "Count"
  )

# 6.3 Relationship between danceability and popularity
ggplot(spotify, aes(x = danceability, y = popularity)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm") +
  theme_minimal() +
  labs(
    title = "Danceability vs Popularity",
    x = "Danceability",
    y = "Popularity"
  )

# 6.4 Relationship between energy and popularity
ggplot(spotify, aes(x = energy, y = popularity)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm") +
  theme_minimal() +
  labs(
    title = "Energy vs Popularity",
    x = "Energy",
    y = "Popularity"
  )

# 6.5 Relationship between valence and popularity
ggplot(spotify, aes(x = valence, y = popularity)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm") +
  theme_minimal() +
  labs(
    title = "Valence vs Popularity",
    x = "Valence",
    y = "Popularity"
  )

# 6.6 Popularity by energy level
ggplot(spotify, aes(x = energy_level, y = popularity)) +
  geom_boxplot() +
  theme_minimal() +
  labs(
    title = "Popularity by Energy Level",
    x = "Energy Level",
    y = "Popularity"
  )

# 6.7 Popularity by danceability level
ggplot(spotify, aes(x = danceability_level, y = popularity)) +
  geom_boxplot() +
  theme_minimal() +
  labs(
    title = "Popularity by Danceability Level",
    x = "Danceability Level",
    y = "Popularity"
  )

# 6.8 Popularity by tempo category
ggplot(spotify, aes(x = tempo_category, y = popularity)) +
  geom_boxplot() +
  theme_minimal() +
  labs(
    title = "Popularity by Tempo Category",
    x = "Tempo Category",
    y = "Popularity"
  )

# -----------------------------
# 7. Correlation analysis
# -----------------------------
# Select numeric variables for correlation matrix
numeric_data <- spotify %>%
  select(
    popularity, danceability, energy, loudness,
    tempo, valence, acousticness, duration_min
  )

# Compute correlation matrix
cor_matrix <- cor(numeric_data, use = "complete.obs")

# Visualize correlation matrix
corrplot(cor_matrix, method = "circle")

# -----------------------------
# 8. Train-test split
# -----------------------------
# This will be used later for predictive models
set.seed(123)

trainIndex <- createDataPartition(spotify$is_hit, p = 0.8, list = FALSE)

train <- spotify[trainIndex, ]
test  <- spotify[-trainIndex, ]

# Check sizes
nrow(train)
nrow(test)

# -----------------------------
# 9. Logistic regression
# -----------------------------
# Predict whether a song is a hit
model_logit <- glm(
  is_hit ~ danceability + energy + loudness + tempo + valence + acousticness + duration_min,
  data = train,
  family = "binomial"
)

summary(model_logit)

# Predict probabilities on test set
pred_probs <- predict(model_logit, test, type = "response")

# Convert probabilities to classes
pred_class <- ifelse(pred_probs > 0.5, "Hit", "Not Hit")
pred_class <- as.factor(pred_class)

# Confusion matrix
confusionMatrix(pred_class, test$is_hit)

# -----------------------------
# 10. Linear regression
# -----------------------------
# Predict song popularity
model_lm <- lm(
  popularity ~ danceability + energy + loudness + tempo + valence + acousticness + duration_min,
  data = train
)

summary(model_lm)

# -----------------------------
# 11. Decision tree
# -----------------------------
# Classification tree for hit prediction
tree_model <- rpart(
  is_hit ~ danceability + energy + loudness + tempo + valence + acousticness + duration_min,
  data = train,
  method = "class"
)

rpart.plot(tree_model)

# -----------------------------
# 12. Optional: Save plots
# -----------------------------
# Example:
# ggsave("outputs/popularity_histogram.png", width = 8, height = 5)