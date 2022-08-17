import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

data = pd.read_csv('./watson_healthcare_modified.csv')

# remove unwanted columns (all dataset is > 18 so irrelevant to classification )
# this is recursive as added more cols to drop having run with all and seen those that have zero or near zero coefficients
data.drop(columns=["EmployeeID", "EmployeeCount", "Over18", "StandardHours", "Education"], inplace=True)

# convert binary category values into binary ( i.e. y / n into 1 0)
data['OverTime'] = data['OverTime'].map({"Yes": 1, "No": 0})
data['Gender'] = data['Gender'].map({"Male": 1, "Female": 0}) # only for this dataset, not generally
data['Attrition'] = data['Attrition'].map({"Yes": 1, "No": 0}) # could use sklearn LabelEncoder here but keeping all data parsing in pandas

# convert 'category' features into "one-hot" equivalent i.e. Department into series of 1,0 columns for each distinct dept name 
# using pandas get_dummies  (but could equally use sklearn OneHotEncoder)
data = pd.get_dummies(data)

# Normalizing data
labels = data.pop('Attrition')

scaler = StandardScaler()
scaler.fit(data)

features = scaler.transform(data)
feature_names = scaler.get_feature_names_out()

# Split data into training and test sets
features_train, features_test, labels_train, labels_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=100
)

# Create Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(features_train, labels_train)
intercept = lr_model.intercept_
coefs = lr_model.coef_

for i in range(len(feature_names)):
    print(f"{feature_names[i]}: {coefs[0, i]:.5f}")

label_predictions = lr_model.predict(features_test)

# View confusion matrix
con_matrix = confusion_matrix(labels_test, label_predictions)
print('\n', con_matrix, '\n')

# View scores
print(f"True Positive: {con_matrix[1][1]}")
print(f"False Positive: {con_matrix[0][1]}")
print(f"True Negative: {con_matrix[0][0]}")
print(f"False Negative: {con_matrix[1][0]}")

print(f"\nAccuracy score: {accuracy_score(labels_test, label_predictions):.2f}")
print(f"Precision score: {precision_score(labels_test, label_predictions):.2f}")
print(f"Recall score: {recall_score(labels_test, label_predictions):.2f}")
print(f"F1 score: {f1_score(labels_test, label_predictions):.2f}")

# Graph True or False predictions vs prediction probability
prediction_probabilities = lr_model.predict_proba(features_test)[:, 1]

# labels_test shows as type pandas.Series but either via [i] or .iloc(i)
# results in KeyError (i) or ValueError(axis = i) respectively.  No-idea why...
# so casting to simple list.
list_labels_test = list(labels_test)

true_false = []
for i in range(len(labels_test)):
    if list_labels_test[i] == label_predictions[i]:
        true_false.append(1)
    else:
        true_false.append(0)

plt.scatter(prediction_probabilities, list_labels_test,alpha=0.3)
plt.show()