import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Load data from CSV file
# Create a new column after the first column in the generated features.csv file to provide the labels of your domain. 
# In our sample features_ADNI2.csv file, its "Manufacturer" column with 'GE', 'Siemens' and 'Philips' labels.
# You can remove the TSNEX, TSNEY, UMAPX, and UMAPY columns from the features.csv file for this feature importance analysis.
data = pd.read_csv("features_ADNI2.csv")

# Separate the labels (class) from the features
labels = data.iloc[:, 1]
features = data.iloc[:, 2:]

# Fit a random forest classifier to the data
rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(features, labels)

# Extract feature importance from the random forest model
importances = rf.feature_importances_

# Sort feature importance in descending order
sorted_indices = np.argsort(importances)[::-1]


# Create a new DataFrame to store the sorted feature importances
sorted_df = pd.DataFrame(columns=['Feature', 'Importance'])

# Add the feature names and importances to the new DataFrame
for i in range(features.shape[1]):
    sorted_df.loc[i] = [features.columns[i], importances[i]]

# Save the sorted feature importances to a CSV file
#sorted_df.to_csv('feature_importances.csv', index=False)

# Print the feature ranking
print("Feature ranking:")
for i, index in enumerate(sorted_indices):
    print("{}) {} ({:.3f})".format(i + 1, features.columns[index], importances[index]))


# Visualize the feature importances
plt.figure()
plt.title("Feature importances for ADNI2")
plt.bar(range(features.shape[1]), importances[sorted_indices])
plt.xticks(range(features.shape[1]), features.columns[sorted_indices], rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Select top n features and plot their importance
n = 10
sfm = SelectFromModel(rf, threshold=-np.inf, max_features=n)
sfm.fit(features, labels)
selected = sfm.transform(features)
indices_selected = np.argsort(sfm.estimator_.feature_importances_)[::-1]
plt.figure()
plt.title("Top {} Feature importances for ADNI2".format(n))
plt.bar(range(n), sfm.estimator_.feature_importances_[indices_selected][:n])
plt.xticks(range(n), features.columns[indices_selected][:n], rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
