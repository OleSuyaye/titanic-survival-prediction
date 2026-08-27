# Loading Relevant Libraries
import pandas as pd
import joblib
from pathlib import path

# Path
MODEL_PATH = "../Models/titanic_model.joblib"
TEST_PATH = "../data/test.csv"
OUTPUT_PATH = "../Submissions/submission.csv"

# Load the model
model = joblib.load(MODEL_PATH)

# Load the test dataset
test_df = pd.read_csv(TEST_PATH)

# Keep passenger IDs
passenger_ids = test_df["PassengerId"]

# Feature Engineering
X_test = test_df[['Pclass','Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']]

# Make predictions
predictions = model.predict(X_test)

# Create submission
submission = pd.DataFrame({
                            "PassengerId" : passenger_ids,
                            "Survived" : predictions
})

# Saving the submissions
submission.to_csv(OUTPUT_PATH, index= False)

print("Submissions saved in", OUTPUT_PATH)
