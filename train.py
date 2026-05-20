import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv('data/student_data.csv')

le_gender = LabelEncoder()
le_parent = LabelEncoder()
le_internet = LabelEncoder()
le_extra = LabelEncoder()
le_result = LabelEncoder()

df['gender'] = le_gender.fit_transform(df['gender'])
df['parent_education'] = le_parent.fit_transform(df['parent_education'])
df['internet_access'] = le_internet.fit_transform(df['internet_access'])
df['extra_classes'] = le_extra.fit_transform(df['extra_classes'])
df['result'] = le_result.fit_transform(df['result'])

joblib.dump(le_gender, 'model/le_gender.pkl')
joblib.dump(le_parent, 'model/le_parent.pkl')
joblib.dump(le_internet, 'model/le_internet.pkl')
joblib.dump(le_extra, 'model/le_extra.pkl')
joblib.dump(le_result, 'model/le_result.pkl')

X = df.drop(['student_id', 'result', 'final_marks'], axis=1)
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, 'model/student_model.pkl')
print("Model saved!")
print("Scikit-learn version:", __import__('sklearn').__version__)