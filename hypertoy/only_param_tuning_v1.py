from os import name
from sys import modules
import pickle
import copy

from hypernets.core.ops import Identity, HyperInput, ModuleChoice
from hypernets.core.search_space import Choice, HyperNode, HyperSpace, ModuleSpace
from hypernets.model import Estimator, HyperModel
from hypernets.utils import fs
from hypernets.searchers.grid_searcher import GridSearcher

from sklearn import neighbors


class Param_space(object):
    def __init__(self, **kwargs):
        super(Param_space, self).__init__()

    @property
    def knn(self):
        return dict(
            cls=neighbors.KNeighborsClassifier,
            n_neighbors=Choice([2, 3, 5, 6]),
            weights=Choice(['uniform', 'distance']),
            algorithm=Choice(['auto', 'ball_tree', 'kd_tree', 'brute']),
            leaf_size=Choice([20, 30, 40]),
            p=Choice([1, 2]),
            metric='minkowski',
            metric_params=None, 
            n_jobs=None,
        )

    def __call__(self, *args, **kwargs):
        space = HyperSpace()

        with space.as_default():
            hyper_input = HyperInput(name='input1')
            model = self.knn
            modules = [ModuleSpace(name=f'{model["cls"].__name__}', **model)]
            outputs = ModuleChoice(modules)(hyper_input)
            space.set_inputs(hyper_input)

        return space

class KnnEstimator(Estimator):
    def __init__(self, space_sample, task='binary'):
        super(KnnEstimator, self).__init__(space_sample, task)

        out = space_sample.get_outputs()[0]
        kwargs = out.param_values
        #why need this?
        kwargs = {key: value for key, value in kwargs.items() if not isinstance(value, HyperNode)}

        cls = kwargs.pop('cls')
        self.model = cls(**kwargs)
        self.cls = cls
        self.model_args = kwargs
    
    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

        return self
    
    def predict(self, X, **kwargs):
        pred = self.model.predict(X, **kwargs)

        return pred

    def evaluate(self, X, y, **kwargs):
        scores = self.model.score(X, y)

        return scores
    
    def save(self, model_file):
        with fs.open(model_file, 'wb') as f:
            pickle.dump(self, f, protocol=4)

    @staticmethod
    def load(model_file):
        with fs.open(model_file, 'rb') as f:
            return pickle.load(f)

    def get_iteration_scores():
        return []

class KnnModel(HyperModel):
    def __init__(self, searcher, reward_metric=None, task=None):
        super(KnnModel, self).__init__(searcher, reward_metric=reward_metric, task=task)
    
    def _get_estimator(self, space_sample):
        return KnnEstimator(space_sample, task=self.task)
    
    def load_estimator(self, model_file):
        return KnnEstimator.load(model_file)

#implementing the search method using Hypernets directly.
def param_search():
    return NotImplementedError

#calling the defined search method of the HyperModel.

#train
def train(X_train, y_train, X_eval, y_eval, optimize_direction='max', **kwargs):

    search_space = Param_space()
    searcher = GridSearcher(search_space, optimize_direction=optimize_direction)
    model = KnnModel(searcher=searcher, task='multiclass', reward_metric='accuracy')
    model.search(X_train, y_train, X_eval, y_eval, **kwargs)
    best_model = model.get_best_trial()
    final_model = model.final_train(best_model.space_sample, X_train, y_train)
    return model, final_model

params = {'n_neighbors': Choice([2, 3, 5, 6]),
            'weights': Choice(['uniform', 'distance']),
            'algorithm': Choice(['auto', 'ball_tree', 'kd_tree', 'brute']),
            'leaf_size': Choice([20, 30, 40]),
            'p': Choice([1, 2]),
        }

def score_function(X_train, y_train, X_evl, y_evl, **params):
    model = neighbors.KNeighborsClassifier(**params)
    model.fit(X_train, y_train)
    scores = model.evaluate(X_evl, y_evl)
    
    return scores