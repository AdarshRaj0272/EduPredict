import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

study = np.round(np.random.uniform(1, 10, n), 1)
attendance = np.round(np.random.uniform(50, 100, n), 1)
previous = np.random.randint(40, 100, n)
final_marks = (study * 4 + attendance * 0.3 + previous * 0.3 + np.random.randint(-5, 5, n)).astype(int)
final_marks = np.clip(final_marks, 35, 100)

data = {
    'student_id': range(1, n+1),
    'age': np.random.randint(15, 20, n),
    'gender': np.random.choice(['Male', 'Female'], n),
    'study_hours_per_day': study,
    'attendance_percent': attendance,
    'previous_marks': previous,
    'parent_education': np.random.choice(['Primary', 'Secondary', 'Graduate', 'Postgraduate'], n),
    'internet_access': np.random.choice(['Yes', 'No'], n),
    'extra_classes': np.random.choice(['Yes', 'No'], n),
    'final_marks': final_marks,
}

df = pd.DataFrame(data)
df['result'] = df['final_marks'].apply(lambda x: 'Pass' if x >= 60 else 'Fail')
df.to_csv('data/student_data.csv', index=False)
print("Dataset ready!")
print(df['result'].value_counts())