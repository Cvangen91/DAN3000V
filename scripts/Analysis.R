library(dplyr)
library(readr)
library(ggplot2)
library(tidyr)
library(caret)
library(corrplot)
library(rpart)
library(rpart.plot)
library(broom)

spotify <- read_delim("data_processed/spotify_clean.csv", delim = ";")

spotify <- spotify %>%
  mutate(across(
    c(acousticness, danceability, energy, instrumentalness,
      liveness, loudness, speechiness, tempo, popularity,
      valence, duration_min),
    ~ as.numeric(gsub(",", ".", .))
  ))

glimpse(spotify)
summary(spotify)

head(spotify)

ggplot(spotify, aes(x = popularity)) +
  geom_histogram(bins = 30) +
  theme_minimal()

ggplot(spotify, aes(x = is_hit)) +
  geom_bar() +
  theme_minimal()

ggplot(spotify, aes(x = danceability, y = popularity)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm") +
  theme_minimal()

numeric_data <- spotify %>%
  select(popularity, danceability, energy, loudness, tempo, valence, acousticness, duration_min)

cor_matrix <- cor(numeric_data)

corrplot(cor_matrix, method = "circle")

