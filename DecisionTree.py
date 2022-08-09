import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('import your csv file here')

#drop inefficient columns
df.drop(df.columns[[0, 3, 17, 18, 19, 20, 21, 22, 23, 24, 31, 32, 33, 42, 49, 55, 56, 57, 58, 59, 60, 73, 74, 75, 76]], axis=1, inplace=True)


x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)



sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



classifier = DecisionTreeClassifier(criterion='entropy' , splitter='random' , max_features='sqrt')
classifier.fit(x_train, y_train)


y_pred = classifier.predict(x_test)


Conf_Matrix = confusion_matrix(y_test, y_pred)
Accuracy = accuracy_score(y_test, y_pred)
Report = classification_report(y_test, y_pred)

print('The accuracy of D_Tree is : ', Accuracy)
print('\n \n')
print('Cofusion_matrix is: \n', Conf_Matrix)
print('\n \n')
print('Classification report is: \n', Report)