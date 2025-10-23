import pandas as pd

# Load your training data
train_data = pd.read_csv("iberfire_train.csv", delimiter=";")

target_column = "is_fire"

# Drop target to compute correlations
corrs = train_data.corr(numeric_only=True)[target_column].sort_values(ascending=False)

print("\nTop features most correlated with target:")
print(corrs.head(15))  # top positive correlations
print("\nTop features most negatively correlated with target:")
print(corrs.tail(15))  # top negative correlations
