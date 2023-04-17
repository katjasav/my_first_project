Датасет был разделен на 80% train и 20% test.

В качестве базовой модели была выбрана LogisticRegression с учетом перебора значений гиперпараметра C, а также валидации на 5 фолдах с помощью GridSearchCV.
В Pipeline добавлена нормировка числовых признаков с помощью StandartScaler, а также предобработка категориальных признаков с помощью OneHotEncoder.
В качестве скоринга использована метрика recall.
Лучший результат модель показала при использовании гиперпараметра C = 100.
По метрике Recall модель логистической регрессии показала неплохие результаты (0.63 на train и 0.64 на test).
По метрике ROC-AUC модель также показала неплохие результаты (0.7536 на train и 0.7572 на test).

Поскольку задача является стандартной в плане выбора того или иного значения признака, а в конечном итоге присвоение одного из бинарных классов, то в первую очередь, был  протестирован классический алгоритм машинного обучения - дерево решений. Также в связи с тем, что модели градиентного бустинга, как правило, показывают достаточно высокие результаты, была протестирована XGBoost, т.к. она по временным параметрам отрабатывает быстрее, чем другие модели градиентного бустинга. Оптимальные гиперпараметры подбирались с помощью GridSearchCV (основной гиперпараметр max_depth для обеих моделей; для Decision Tree также включены в подбор параметров различные критерии классификации).

Decision Tree с учетом перебора значений гиперпараметров criterion, max_depth, а также валидации на 5 фолдах с помощью GridSearchCV.
В Pipeline добавлена нормировка числовых признаков с помощью StandartScaler. Предобработка категориальных признаков с помощью OneHotEncoder опущена, поскольку модель хорошо работает с категориальными признаками.
В качестве скоринга использована метрика recall.
Лучший результат модель показала при использовании гиперпараметров: 'criterion': 'gini', 'max_depth': 20.
Наиболее значимыми признаками оказались уровень сахара и индекс массы тела. Пол и возраст оказались не значимыми признаками.
По метрике Recall модель показала практически идеальные результаты (1.0 на train и 0.9989 на test).
По метрике ROC-AUC модель также показала отличные результаты (0.9991 на train и 0.9975 на test).

XGBoost с учетом перебора значений гиперпараметра max_depth, а также валидации на 5 фолдах с помощью GridSearchCV.
В Pipeline добавлена нормировка числовых признаков с помощью StandartScaler, а также предобработка категориальных признаков с помощью OneHotEncoder.
В качестве скоринга использована метрика recall.
Лучший результат показала при использовании гиперпараметра 'max_depth': 6.
По метрике Recall модель показала идеальные результаты (1.0 на train и 1.0 на test).
По метрике ROC-AUC модель также показала отличные результаты (1.0 на train и 1.0 на test).

Качество работы Decision Tree по метрике Recall составляет 1.0 на train и 0.9989 на test, а XGBoost - 1.0 и на train, и на test.
Качество работы Decision Tree по метрике Roc-AUC составляет 0.9991 на train и 0.9975 на test, а XGBoost - 1.0 и на train, и на test.
Время обучения Decision Tree намного меньше (2,67 сек) по сравнению с XGBoost (24,8 сек).
Поскольку разница в качестве незначительная (0.1% по Recall, 0,1-0,25% по ROC-AUC), а время работы Decision Tree выше + не требуется дополнительная обработка категориальных признаков, то остановимся на выборе данной модели.

DEEP LEARNING
Полносвязная нейронная сеть развернута с помощью PyTorch.
Данные преобразованы в тензоры и разделены на батчи размером 200 (при создании DataLoader перемешаны на тренировочном датасете).
В качестве скора использована recall из torchmetrics.
Развернута полносвязная сеть, состоящая из входного слоя, трех скрытых слоев и выходного слоя.
Количество нейронов подобрано опытным путем в результате нескольких тестов.
Добавлена оптимизация с помощью BatchNorm1d в каждом слое.
В качестве функции активации использована сигмоида.
В качестве критерия использована кросс-энтропия.
В качестве оптимизатора использован Adam с учетом регуляризации по весам, а также изменяемой скорости обучения - ExponentialLR.
Модель обучена на 30 эпохах.
При обучении на 30 эпохах с учетом примененных оптимизаций удалось достичь качества по Recall: 0.6514 на train и 0.6459 на test. Модель недообучилась. Качество метрики растет медленно.

Модель протестирована на различных функциях активации: Sigmoid, ReLu, LeakyReLu.
Лучше всего отработала модель с активацией ReLU.

Модель протестирована с учетом активации ReLu на 100 эпохах.
По метрике Recall модель логистической регрессии показала неплохие результаты (0.7865 на train и 0.7826 на test).
По метрике ROC-AUC модель также показала неплохие результаты (0.8459 на train и 0.8370 на test).
На 100 эпохах модель не дообучилась, но обучение очень медленное (показатели растут медленно).
Возможно, необходимо выбрать иную архитектуру, либо выполнить дополнительную оптимизацию параметров.

Вывод: оптимальным вариантом оказалась модель Decision Tree с учетом гиперпараметров: 'criterion': 'gini', 'max_depth': 20.

P.s. более подробный отчет с учетом визуализации данных и результатов тестирования моделей можно увидеть в файле Stroke_prediction_SavinaEA.ipynb
