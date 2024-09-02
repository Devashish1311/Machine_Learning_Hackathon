import numpy as np
import pandas as pd

df = pd.read_csv("C:/Users/DELL/Downloads/netflix_titles.csv/netflix_titles.csv")
# Display the first few rows of the dataset
print(df.head())

# Display basic information about the dataset
print(df.info())

# Summary statistics of the dataset
print(df.describe(include='all'))
import matplotlib.pyplot as plt
import seaborn as sns

# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)

# Visualizing missing values using a heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in the Dataset')
plt.show()

# Visualizing data distribution for 'release_year'
plt.figure(figsize=(8,5))
sns.histplot(df['release_year'], bins=30, kde=True)
plt.title('Distribution of Release Year')
plt.xlabel('Release Year')
plt.ylabel('Frequency')
plt.show()

# Visualizing data distribution for 'type'
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='type')
plt.title('Distribution of Content Type')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()
# Question time:
# Question 1: What is the distribution of Netflix content over the years? Has Netflix's production increased over time?
# Question 2: What genres are most common in Netflix's catalog?
# Question 3: What is the distribution of content types (movies vs. TV shows)? Are certain content types more popular in specific years?
# Question 4: What type of genres have had increase in production and which have lost the watch battle ?
# Handling missing values
df.fillna({'director':'Unknown'}, inplace=True)
df.fillna({'cast':'Unknown'}, inplace=True)
df.fillna({'country':'Unknown'}, inplace=True)
df.dropna(subset=['date_added'], inplace=True)

# Convert categorical data using one-hot encoding if necessary
df_encoded = pd.get_dummies(df, columns=['type', 'rating'])

print(df_encoded.head())
# Summary statistics for numerical data
print(df['release_year'].describe())

# Fill NaN values with an empty string or a placeholder
df['duration'] = df['duration'].fillna('Unknown')

# Ensure all values are strings
df['duration'] = df['duration'].astype(str)

# Extract numeric duration for movies and handle TV shows separately
df['duration_numeric'] = df['duration'].str.extract('(\d+)').astype(float)

# Create a new column 'duration_type' to differentiate between minutes and seasons
df['duration_type'] = df['duration'].apply(lambda x: 'min' if 'min' in x else 'seasons')

# Now we plot again, but only for movies
movies_df = df[df['type'] == 'Movie']
plt.figure(figsize=(10, 6))
sns.scatterplot(data=movies_df, x='release_year', y='duration_numeric', hue='type')
plt.title('Release Year vs. Duration for Movies')
plt.xlabel('Release Year')
plt.ylabel('Duration (Minutes)')
plt.show()

# We do a separate plot for TV Shows
tv_shows_df = df[df['type'] == 'TV Show']
plt.figure(figsize=(10, 6))
sns.scatterplot(data=tv_shows_df, x='release_year', y='duration_numeric', hue='type')
plt.title('Release Year vs. Duration for TV Shows')
plt.xlabel('Release Year')
plt.ylabel('Number of Seasons')
plt.show()

# Grouping data by release year and type
content_by_year = df.groupby(['release_year', 'type']).size().unstack().fillna(0)
print(content_by_year)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def extract_duration(x):
    try:
        if 'min' in x:
            return int(x.split(' ')[0])
        elif 'Season' in x:
            return int(x.split(' ')[0]) * 60  # Convert seasons to minutes
    except ValueError:
        return np.nan
    return np.nan

df['duration_numeric'] = df['duration'].apply(extract_duration)

# Create a new column 'duration_type' to differentiate between minutes and seasons
df['duration_type'] = df['duration'].apply(lambda x: 'min' if 'min' in x else 'seasons' if 'Season' in x else 'unknown')

# Drop rows where 'duration_numeric' is NaN if you don't want to include them in the analysis
df = df.dropna(subset=['duration_numeric'])

# Check for errors
print(df[['duration', 'duration_numeric']].head())

# Process 'duration' column
df['duration_numeric'] = df['duration'].apply(lambda x: int(x.split(' ')[0]) if 'min' in x else int(x.split(' ')[0])*60)

# Check the result
print(df)

# Predicting the duration of content based on the release year
X = df[['release_year']]
y = df['duration_numeric']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Plotting the regression line
plt.figure(figsize=(10,6))
sns.regplot(x=y_test, y=y_pred, line_kws={"color":"red"})
plt.title('Actual vs. Predicted Duration')
plt.xlabel('Actual Duration')
plt.ylabel('Predicted Duration')
plt.show()
