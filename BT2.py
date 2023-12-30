from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load tập dữ liệu Iris
iris = load_iris()
X, y = iris.data, iris.target

#Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)
accuray_dt = accuracy_score(y_test, y_pred_dt)
print("Độ chính xác của Decision Tree: ", accuray_dt)

#Bagging
BaggingClassifier = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
BaggingClassifier.fit(X_train, y_train)
y_pred_bagging = BaggingClassifier.predict(X_test)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
print("Độ chính xác cũa Bagging", accuracy_bagging)

#AdaBoost
Adaboost_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=42)
Adaboost_classifier.fit(X_train, y_train)
y_pred_adaboost = Adaboost_classifier.predict(X_test)
accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
print("Độ chính xác của AdaBoost: ", accuracy_adaboost)

#Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Độ chính xác của Random Forest: ", accuracy_rf)
