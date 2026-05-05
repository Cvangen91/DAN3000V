# Analysis.R
# Spotify Data Analytics Project
# This script explores the cleaned Spotify dataset and builds models
# to analyse whether audio features and genre can predict song popularity.

# -----------------------------
# 1. Load required packages
# -----------------------------
library(dplyr)
library(readr)
library(tidyr)
library(caret)
library(rpart)
library(rpart.plot)
library(broom)

# -----------------------------
# 2. Load dataset
# -----------------------------
spotify <- read_csv("data_processed/spotify_clean_new - without comma.csv")

# -----------------------------
# 3. Inspect data
# -----------------------------
glimpse(spotify)
summary(spotify)

# -----------------------------
# 4. Exploratory Data Analysis
# -----------------------------

# Check hit / non-hit distribution
table(spotify$is_hit)
prop.table(table(spotify$is_hit))

# Identify top 10 genres by average popularity
top_genres <- spotify %>%
  group_by(track_genre) %>%
  summarise(avg_popularity = mean(popularity, na.rm = TRUE)) %>%
  arrange(desc(avg_popularity)) %>%
  slice_head(n = 10)

top_genres

# Simple linear regressions for EDA relationships
eda_models <- list(
  danceability = lm(popularity ~ danceability, data = spotify),
  loudness = lm(popularity ~ loudness, data = spotify),
  valence = lm(popularity ~ valence, data = spotify),
  instrumentalness = lm(popularity ~ instrumentalness, data = spotify),
  speechiness = lm(popularity ~ speechiness, data = spotify)
)

eda_results <- bind_rows(
  lapply(names(eda_models), function(variable_name) {
    tidy(eda_models[[variable_name]]) %>%
      filter(term != "(Intercept)") %>%
      mutate(variable = variable_name)
  })
) %>%
  select(variable, estimate, p.value)

eda_results

# -----------------------------
# 5. Prepare modelling datasets
# -----------------------------

# Dataset with all genres
spotify_model_all <- spotify %>%
  select(
    popularity,
    is_hit,
    danceability,
    loudness,
    instrumentalness,
    speechiness,
    valence,
    track_genre,
    duration_min
  ) %>%
  mutate(
    track_genre = as.factor(track_genre),
    is_hit = factor(
      is_hit,
      levels = c("Not Hit", "Hit"),
      labels = c("No", "Yes")
    )
  ) %>%
  drop_na()

# Dataset with top 10 genres by average popularity
spotify_model_top10 <- spotify_model_all %>%
  filter(track_genre %in% top_genres$track_genre) %>%
  mutate(track_genre = droplevels(track_genre))

# Check modelling datasets
table(spotify_model_top10$track_genre)
table(spotify_model_all$is_hit)

# -----------------------------
# 6. Cross-validation setup
# -----------------------------

set.seed(123)

classification_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

regression_control <- trainControl(
  method = "cv",
  number = 5
)

# -----------------------------
# 7. Logistic Regression - Top 10 Genres by Average Popularity Model
# -----------------------------

model_logit_top10 <- train(
  is_hit ~ danceability + loudness + instrumentalness +
    speechiness + valence + track_genre,
  data = spotify_model_top10,
  method = "glm",
  family = "binomial",
  trControl = classification_control,
  metric = "ROC"
)

model_logit_top10

predictions_top10 <- predict(model_logit_top10, spotify_model_top10)
confusionMatrix(predictions_top10, spotify_model_top10$is_hit)

summary(model_logit_top10$finalModel)

# -----------------------------
# 8. Logistic Regression - All Genres Model
# -----------------------------

model_logit_all <- train(
  is_hit ~ danceability + loudness + instrumentalness +
    speechiness + valence + track_genre,
  data = spotify_model_all,
  method = "glm",
  family = "binomial",
  trControl = classification_control,
  metric = "ROC"
)

model_logit_all

predictions_all <- predict(model_logit_all, spotify_model_all)
confusionMatrix(predictions_all, spotify_model_all$is_hit)

summary(model_logit_all$finalModel)

# -----------------------------
# 9. Linear Regression Model
# -----------------------------

model_lm <- train(
  popularity ~ danceability + loudness + instrumentalness +
    speechiness + valence + track_genre,
  data = spotify_model_all,
  method = "lm",
  trControl = regression_control
)

model_lm
summary(model_lm$finalModel)

# -----------------------------
# 10. Decision Tree Model
# -----------------------------

tree_model <- rpart(
  is_hit ~ danceability + loudness + instrumentalness +
    speechiness + valence + track_genre,
  data = spotify_model_all,
  method = "class"
)

rpart.plot(
  tree_model,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  cex = 0.7,
  split.fun = function(x, labs, digits, varlen, faclen) {
    ifelse(grepl("track_genre", labs), "Genre group split", labs)
  }
)

tree_predictions <- predict(tree_model, spotify_model_all, type = "class")
confusionMatrix(tree_predictions, spotify_model_all$is_hit)