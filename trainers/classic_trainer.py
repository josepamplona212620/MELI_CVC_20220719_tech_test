from base.base_trainer import BaseTrain
from models.model_switcher import ModelSwitcher
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import pickle

class ClasicKfoldCrossValidationTrainer(BaseTrain):
    def __init__(self, model, data, config, splits, seed):
        super(ClasicKfoldCrossValidationTrainer, self).__init__(model, data, config)
        self.acc = []
        self.params = []

        self.steps = Pipeline([('scaler', StandardScaler()),
                               ('model', ModelSwitcher())])
        self.skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        self.cv = GridSearchCV(self.steps, self.config, cv=self.skf)

    def train(self):
        self.cv.fit(self.train_data, self.test_data)
        self.acc = self.cv.best_score_
        self.params = self.cv.best_params_

    def best_model_save(self, location):
        pickle.dump(self.cv.best_estimator_, open(location, 'wb'))
