from os import name
from sys import modules
import pickle

from hypernets.core.ops import HyperInput, ModuleChoice
from hypernets.core.search_space import Choice, HyperNode, HyperSpace, ModuleSpace
from hypernets.model import Estimator, HyperModel
from hypernets.utils import fs

from sklearn import neighbors

def param_space():
    space = HyperSpace()

    model_param = dict(
            n_neighbors=Choice([2, 3, 5, 6]),
            weights=Choice(['uniform', 'distance']),
            algorithm=Choice(['auto', 'ball_tree', 'kd_tree', 'brute']),
            leaf_size=Choice([20, 30, 40]),
            p=Choice([1, 2]),
            metric='minkowski',
            metric_params=None, 
            n_jobs=None,
        )

    with space.as_default():
        hyper_input = HyperInput(name='input1')
        model = neighbors.KNeighborsClassifier
        modules = ModuleSpace(neighbors.KNeighborsClassifier, *model_param)
        outputs = ModuleChoice(modules)(hyper_input)
        space.set_inputs(hyper_input)

    return space


class KnnEstimator(Estimator):
    def __init__(self, space_sample, task='binary'):
        super(KnnEstimator, self).__init__(space_sample, task)

        out = space_sample.get_outputs()[0]
        kwargs = out.param_values
        kwargs = {key: value for key, value in kwargs.items() if not isinstance(value, HyperNode)}

        self.model = neighbors.KNeighborsClassifier(**kwargs)
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