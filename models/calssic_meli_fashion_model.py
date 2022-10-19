from base.base_model import BaseModel
from models.model_switcher import ModelSwitcher
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

steps = Pipeline([('scaler', StandardScaler()),
                  ('model', ModelSwitcher())])

parameters = [
    {
        'clf__estimator': [SGDClassifier()], # SVM if hinge loss / logreg if log loss
        'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
        'tfidf__stop_words': ['english', None],
        'clf__estimator__penalty': ('l2', 'elasticnet', 'l1'),
        'clf__estimator__max_iter': [50, 80],
        'clf__estimator__tol': [1e-4],
        'clf__estimator__loss': ['hinge', 'log', 'modified_huber'],
    },
    {
        'clf__estimator': [MultinomialNB()],
        'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
        'tfidf__stop_words': [None],
        'clf__estimator__alpha': (1e-2, 1e-3, 1e-1),
    },
]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

cv = GridSearchCV(pipeline, parameters)

cv.fit(X_train, y_train)

knn_scaled = pipeline.fit(X_train, y_train)


class SimpleMnistModel(BaseModel):
    def __init__(self, config):
        super(SimpleMnistModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_shape=(28 * 28,)))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=self.config.model.optimizer,
            metrics=['acc'],
        )
