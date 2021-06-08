"""

"""
import pickle
import re

from sklearn import pipeline as sk_pipeline

#from hypergbm.pipeline import ComposeTransformer
from hypernets.model.estimator import Estimator
from hypernets.model.hyper_model import HyperModel
from hypernets.tabular.cache import cache
from hypernets.tabular.data_cleaner import DataCleaner
from hypernets.tabular.metrics import calc_score
from hypernets.utils import fs
from .estimator import HyperEstimator

class toy_KNN_estimator(Estimator):
    """

    """
    #need more functions dealing with the tabular data, such as fit_transform_data
    def __init__(self, task, space_sample, data_cleaner_params=None):
        """
        params: 
            data_pipeline:
            data_cleaner_params:
            data_cleaner:
            pipeline_signature:
            fit_kwargs:
            class_balancing:
            _build_model:
        """
        super(toy_KNN_estimator, self).__init__(space_sample=space_sample, task=task)    
        self.data_pipeline = None
        self.data_cleaner_params = data_cleaner_params
        self.data_cleaner = None
        self.knn_model = None
        self.pipeline_signature = None
        self.fit_kwargs = None
        self.class_balancing = None
        self.classes_ = None
        self._build_model(space_sample)

        return 

    def _build_model(self, space_sample):
        """This function builds a kNN model, space_sample should be a..., the
        compile_and_forward, which is used to..., comes from ..."""
        space, _ = space_sample.compile_and_forward()

        """get_outputs():this one is used to... and comes from"""
        outputs = space.get_outputs()
        assert len(outputs) == 1, 'The space can only contains 1 output.'
        assert isinstance(outputs[0], HyperEstimator), 'The output of space must be `HyperEstimator`.'
        if outputs[0].estimator is None:
            outputs[0].build_estimator(self.task)
        self.knn_model = outputs[0].estimator
        self.class_balancing = outputs[0].class_balancing
        self.fit_kwargs = outputs[0].fit_kwargs

        """dealing with the pipeline"""

    def build_pipeline(self, space, last_transformer):
        raise NotImplementedError

    def fit_transform_data(self, X, y):
        if self.data_cleaner is not None:
            X, y = self.data_cleaner.fit_transform(X, y)
        X = self.data_pipeline.fit_transform(X, y)
        return X

    def transform_data(self, X, y):
        if self.data_cleaner is not None:
            X = self.data_cleaner.transform(X)
        X = self.data_pipeline.transform(X)
        return  X

    def get_iteration_scores(self):
        iteration_scores = {}

        def get_scores(knn_model, iteration_scores):
            if hasattr(knn_model, 'iteration_scores'):
                if knn_model.__dict__.get('group_id'):
                    group_id = knn_model.group_id
                else:
                    group_id = knn_model.__class__.__name__
                iteration_scores[group_id] = knn_model.iteration_scores

        get_scores(self.knn_model, iteration_scores)
        return iteration_scores

    def fit(self, X, y, **kwargs):
        X = self.fit_transform_data(X, y)

        eval_set = kwargs.pop('eval_set', None)
        kwargs = self.fit_kwargs
        if eval_set is None:
            eval_set = kwargs.get('eval_set')
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                X_eval, y_eval = eval_set
                X_eval = self.transform_data(X_eval)
                kwargs['eval_set'] = [(X_eval, y_eval)]
            elif isinstance(eval_set, list):
                es = []
                for i, eval_set_ in enumerate(eval_set):
                    X_eval, y_eval = eval_set_
                    X_eval = self.transform_data(X_eval)
                    es.append((X_eval, y_eval))
                    kwargs['eval_set'] = es

        self.knn_model.group_id = f'{self.knn_model.__class__.__name__}' #
        self.knn_model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        X = self.transform_data(X)
        pred = self.knn_model.predict(X)
        return pred

    def predict_proba(self, X, **kwargs):
        X = self.transform_data(X)
        pred = self.knn_model.predict_proba(X, **kwargs)
        return pred

    def evaluate(self, X, y, metrics='accuracy', **kwargs):
        if self.task != 'regression':
            proba = self.predict_proba(X)
        else:
            proba = None
        preds = self.predict(X)
        scores = calc_score(y, preds, proba, metrics, self.task)
        return scores
    
    def save(self, model_file):
        with fs.open(f'{model_file}', 'wb') as output:
            pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load(model_file):
        with fs.open(f'{model_file}', 'rb') as input:
            model = pickle.load(input)
            return model

    def __getstate__(self):
        raise NotImplementedError

class toy_KNN(HyperModel):
    """
    KNN as a toy example. This class includes several methods as explained below.
    _get_estimator: return a knn estimator
    load_estimatro: load previously saved model
    """

    def __init__(self, searcher, dispatcher=None, callbacks=None, reward_metric='accuracy', task=None,
                 discriminator=None, data_cleaner_params=None):
        self.data_cleaner_params = data_cleaner_params

        HyperModel.__init__(self, searcher, dispatcher=dispatcher, callbacks=callbacks, reward_metric=reward_metric,
                            task=task, discriminator=discriminator)

    def _get_estimator(self, space_sample):
        estimator = toy_KNN_estimator(task=self.task, space_sample=space_sample,
                                      data_cleaner_params=self.data_cleaner_params)
        return estimator

    def load_estimator(self, model_file):
        assert model_file is not None
        return toy_KNN_estimator.load(model_file)