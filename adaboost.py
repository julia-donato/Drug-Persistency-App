import pandas as pd
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split

train = pd.read_csv('cleaned_dataset_final.csv')

X = train.drop('Persistent',axis=1)
y = train['Persistent']

# Feature selection
selector = SelectKBest(chi2, k=5) # select the top 10 features based on chi squared test
selector.fit(X, y)

selected_feature_indices = selector.get_support(indices=True)

selected_feature_scores = selector.scores_[selected_feature_indices]

selected_feature_names = X.columns[selected_feature_indices]

# Keep only the selected features
X = pd.DataFrame(selector.transform(X), columns=selected_feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=1000, learning_rate=1.0)
ada.fit(X_train, y_train)

# Print the top features and their scores
top_features = pd.DataFrame({'Feature': selected_feature_names, 'Score': selected_feature_scores}).sort_values(by='Score', ascending=False).reset_index(drop=True)
print(top_features)

# Save the model
pickle.dump(ada, open('ada_model.pkl','wb'))
