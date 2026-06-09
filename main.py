import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset load
data = pd.read_csv("train.csv.csv")

# Selected features
features = [
    "OverallQual",
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "TotRmsAbvGrd",
    "GarageCars",
    "YearBuilt"
]

# Missing values remove
data = data[features + ["SalePrice"]].dropna()

X = data[features]
y = data["SalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
score = r2_score(y_test, predictions)

print("R2 Score:", score)

# Example prediction
new_house = pd.DataFrame( [[
    8,      # Overall Quality
    2000,   # Living Area
    3,      # Bedrooms
    2,      # Full Bathrooms
    7,      # Total Rooms
    2,      # Garage Capacity
    2015    # Year Built
]],
columns=features
)

predicted_price = model.predict(new_house)

print("Predicted House Price:", predicted_price[0])