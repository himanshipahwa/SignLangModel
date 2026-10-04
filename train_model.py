import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load the data
print("Loading data...")
df = pd.read_csv("hand_data.csv")

# Separate features (the X, Y coordinates) and the label (the sign name)
X = df.drop(columns=["label"])
y = df["label"]

# 2. Split the data
# We use 80% to train the brain and 20% to test if it actually learned
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create and Train the Model
print("Training the ML brain... please wait...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. Test the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Training complete! Accuracy: {accuracy * 100:.2f}%")

# 5. Save the "Brain" to a file
with open("model.p", "wb") as f:
    pickle.dump(model, f)

print("Model saved as 'model.p'. You can now use it for real-time prediction!")