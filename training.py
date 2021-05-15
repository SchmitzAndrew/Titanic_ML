import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


training_file_path = "data/train.csv"
testing_file_path = "data/test.csv"

training_data = pd.read_csv(training_file_path)
testing_data = pd.read_csv(testing_file_path)

y_train  = training_data.Survived

print(testing_data.describe().columns)
y_test = training_data.Survived #check this, should be in test file

titanic_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
X_train = training_data[titanic_features]

X_train.head()

# Training
clf = DecisionTreeClassifier
clf = clf.fit(X_train, y_train)

y_prediction = clf.predict(X_train)

# Results
print("Accuracy:", metrics.accuracy_score(y_test, y_prediction))
