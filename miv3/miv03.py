from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import metrics

data = make_moons(n_samples=10000, noise=0.4)

X = data[0]
y = data[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X0 = X[y == 0]
X1 = X[y == 1]
# plt.plot(X0[:, 0], X0[:, 1], 'r^')
# plt.plot(X1[:, 0], X1[:, 1], 'g*')
# plt.show()

# task3
dec_clf = tree.DecisionTreeClassifier(max_depth=3)  # criterion gini by default
dec_clf2 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=7)
# task4
rnd_clf = RandomForestClassifier(n_estimators=10)
rnd_clf.fit(X_train, y_train)  # for voting

rnd_clf2 = RandomForestClassifier(n_estimators=15)
# task5
log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)

svm = SVC()
svm.fit(X_train, y_train)

# task6
voting_clf = VotingClassifier(estimators=[('sv', svm), ('lr', log_clf), ('rf', rnd_clf)], voting='hard')
list_of_class = [dec_clf, dec_clf2, rnd_clf, rnd_clf2, voting_clf]
for cls in list_of_class:
    cls.fit(X_train, y_train)
    pred = cls.predict(X_test)
    print(cls, "accuracy: ", metrics.accuracy_score(y_test, pred))
    print(metrics.confusion_matrix(y_test, pred))
