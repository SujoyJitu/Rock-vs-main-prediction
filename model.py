import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# ১. ফাইল লোড করা
sonar_data = pd.read_csv('sonar_data.csv.txt', header=None)

# ২. ড্রপ করার সময় axis=1 সরিয়ে দিন
X = sonar_data.drop(columns=60) 
y = sonar_data[60]

# ৩. স্প্লিট করা
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1)

# ৪. মডেল ট্রেন করা
model = LogisticRegression()
model.fit(X_train, y_train)

# ৫. একুরেসি চেক করা
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)
print('Accuracy on training data:', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)
print('Accuracy on test data:', test_data_accuracy)

# ৬. মডেল সেভ করা
joblib.dump(model, 'sonar_model.pkl')
print("Model saved as sonar_model.pkl")