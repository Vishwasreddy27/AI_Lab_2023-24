# Ex.No: 13 Learning – Use Supervised Learning  
### DATE: 31.10.2025                                                                            
### REGISTER NUMBER : 212222060111
### AIM: 
To write a program to train the classifier for -----------------.
###  Algorithm:
1.Data Loading and Preprocessing
2.Feature Selection or Dimensionality Reduction
3.Clustering
4.Visualization and Interpretation
5.Anomaly or Outlier Detection
6.Insights and Reporting

### Program:
```
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('/content/final(2).csv')
df.head()
df.columns
# Dataset shape and column info
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
df.info()
import seaborn as sns
import matplotlib.pyplot as plt

# Plot value counts of target column
sns.countplot(data=df, x='Prediction')
plt.title("Target Class Distribution")
plt.show()
# Plot top 10 frequent 'Threats'
df['Threats'].value_counts().nlargest(10).plot(kind='bar', title='Top Threats')
plt.show()
# Correlation heatmap for numerical columns
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
from sklearn.preprocessing import LabelEncoder

# Drop non-informative columns
df = df.drop(['Time', 'SeddAddress', 'ExpAddress', 'IPaddress'], axis=1)

# Label encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])
# Features and target
X = df.drop('Prediction', axis=1)
y = df['Prediction']
from sklearn.model_selection import train_test_split

# Split data 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report

# Train LightGBM classifier
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))
# Plot feature importance
lgb.plot_importance(model, max_num_features=10)
plt.title("Top 10 Important Features")
plt.show()
```

### Output:

<img width="1235" height="258" alt="image" src="https://github.com/user-attachments/assets/d0667ddc-2583-433c-bf01-22f8940b6157" />
<img width="209" height="412" alt="image" src="https://github.com/user-attachments/assets/782eea4b-02c1-447e-b375-24695cf795b1" />
<img width="442" height="479" alt="image" src="https://github.com/user-attachments/assets/59f7255f-cd50-4ffc-98a2-344244f90d00" />
<img width="589" height="455" alt="image" src="https://github.com/user-attachments/assets/c1c84569-d338-4212-83ce-b400a9d8c50a" />
<img width="569" height="537" alt="image" src="https://github.com/user-attachments/assets/d00028a4-3b63-4649-8bdf-7f3bb6de8b82" />
<img width="600" height="435" alt="image" src="https://github.com/user-attachments/assets/8383e44f-97e3-496f-afcb-53485b3674d3" />
<img width="685" height="287" alt="image" src="https://github.com/user-attachments/assets/b8404608-a18d-48a1-b1fc-972bbfd903cf" />
<img width="658" height="455" alt="image" src="https://github.com/user-attachments/assets/9ec3381f-fc59-4226-93b7-bbd5addac8b6" />

### Result:
Thus the system was trained successfully and the prediction was carried out.
