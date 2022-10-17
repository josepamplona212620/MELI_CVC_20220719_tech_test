from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

def get_grid_search_parámeters(seed):
    parameters = [
        {
            'model__estimator': [SGDClassifier()],  # SVM if hinge loss, logreg if log
            'model__estimator__penalty': ['l2', 'elasticnet', 'l1'],
            'model__estimator__max_iter': [100],
            'model__estimator__tol': [1e-4],
            'model__estimator__loss': ['hinge', 'log', 'modified_huber'],
            'model__estimator__fit_intercept': [False],
            'model__estimator__random_state': [seed],
        },
        {
            'model__estimator': [MLPClassifier()],
            'model__estimator__hidden_layer_sizes': [(40, 20), (60, 30)],
            'model__estimator__solver': ['adam', 'sgd'],
            'model__estimator__alpha': [0.0001, 0.001],
            'model__estimator__learning_rate_init': [0.0001, 0.001],
            'model__estimator__random_state': [seed],
            'model__estimator__max_iter': [100],
        },
        {
            'model__estimator': [RandomForestClassifier()],
            'model__estimator__n_estimators': [50, 100],
            'model__estimator__criterion': ['gini', 'entropy'],
            'model__estimator__max_features': ['sqrt', 'log2'],
            'model__estimator__min_samples_leaf': [0.01, 0.001],
            'model__estimator__random_state': [seed],
        },
    ]
    return parameters