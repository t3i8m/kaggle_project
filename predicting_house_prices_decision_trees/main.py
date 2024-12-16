import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


data = pd.read_csv("predicting_house_prices_decision_trees\melb_data.csv")
print(data.describe())

data_cleaned = data.dropna(axis = 0)
print(data.describe())

Y = data_cleaned.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data_cleaned[melbourne_features]
print(X.describe())

model = DecisionTreeRegressor(random_state=42)
print(X.head())
model.fit(X,Y)
print(model.predict(X.head()))

predicted_home_prices = model.predict(X)
print(mean_absolute_error(Y, predicted_home_prices))


train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))