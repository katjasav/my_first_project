import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import recall_score, roc_auc_score, roc_curve

seed = 17
# деление выборки на test (20%) и train (80%)
X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], 
                                                       train_size=0.8, 
                                                       random_state=seed)
# конвейер подготовки данных числового типа
numeric_transformer = make_pipeline(StandardScaler())

# конвейер подготовки данных категориального типа
preprocessor = ColumnTransformer(
    [
        ('num', numeric_transformer, num_cols)
    ],
    remainder="passthrough",
    verbose_feature_names_out = False,
)
# подготовка и обучение алгоритма
alg = Pipeline(steps=[
    ('enc', preprocessor), # подготовка
    ('alg', tree.DecisionTreeClassifier(random_state=seed)) # алгоритм
])
parameters = {
    # название этпапа подготовки__гиперпараметр: [сетка (список) параметров]
    "alg__max_depth": [2,4,6,10,15,20],
    "alg__criterion": ['gini', 'entropy', 'log_loss']
}
# сетка гиперпараметров
gs = GridSearchCV(alg, # выбор алгоритма или пайплайна (подготовка + алгоритм)
                  parameters, # сетка гиперпараметров
                  cv=5, # количество фолдов кросс-валидации
                  scoring='recall',
                  verbose=2, # отображать процесс перебора гиперпараметров
                  n_jobs=-1) # параллелизм вычислений

gs.fit(X_train, y_train)
print(gs.best_params_)
print(gs.cv_results_)
print(gs.best_score_)

# важность признаков
feat_list = gs.best_estimator_[-2].get_feature_names_out()
imp_list = gs.best_estimator_.named_steps["alg"].feature_importances_
imp_list = [round(i, 4) for i in imp_list]
print(pd.DataFrame(imp_list, index=feat_list, columns=['importance']).sort_values(by='importance', ascending=False))

# показатели эффективности модели
y_train_pred = gs.predict(X_train)
y_test_pred = gs.predict(X_test)
train_recall = recall_score(y_train, y_train_pred)
test_recall = recall_score(y_test, y_test_pred)
print(f'Recall на train: {round(train_recall, 4)}, Recall на test: {round(test_recall, 4)}')

y_train_predicted = gs.predict_proba(X_train)[:, 1]
y_test_predicted = gs.predict_proba(X_test)[:, 1]
train_auc = roc_auc_score(y_train, y_train_predicted)
test_auc = roc_auc_score(y_test, y_test_predicted)

plt.figure(figsize=(10,7))
plt.plot(*roc_curve(y_train, y_train_predicted)[:2], label='train AUC={:.4f}'.format(train_auc))
plt.plot(*roc_curve(y_test, y_test_predicted)[:2], label='test AUC={:.4f}'.format(test_auc))
legend_box = plt.legend(fontsize='large', framealpha=1).get_frame()
legend_box.set_facecolor("white")
legend_box.set_edgecolor("black")
plt.plot(np.linspace(0,1,100), np.linspace(0,1,100))
plt.show()
