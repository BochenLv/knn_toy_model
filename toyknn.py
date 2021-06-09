"""

"""
import pickle
import re

from sklearn import pipeline as sk_pipeline
from pipeline import ComposeTransformer
from estimator import HyperEstimator

from hypernets.model.estimator import Estimator
from hypernets.model.hyper_model import HyperModel
from hypernets.tabular.cache import cache
from hypernets.tabular.data_cleaner import DataCleaner
from hypernets.tabular.metrics import calc_score
from hypernets.utils import fs

class toy_KNN_estimator(Estimator):
    """
    Using this class allows searching for mutiple models by modifying the searching space
    with mutiple models accordingly rather than only including knnEstimator defined in
    estimator.
    """
    #need more functions dealing with the tabular data, such as fit_transform_data
    def __init__(self, task, space_sample, data_cleaner_params=None):
        """
        params: 
            space_sample: a sampled search space returned by the searcher
            data_pipeline: returned by calling self.build_pipeline
            data_cleaner_params: whether performing data cleaning
            data_cleaner: not None if data_cleaner_params is not None
            pipeline_signature: not clear about this function, thus delete for now
            fit_kwargs: 
            class_balancing: whether performing class balancing, bool
            _build_model: as the name implied, establishing the model
        """
        super(toy_KNN_estimator, self).__init__(space_sample=space_sample, task=task)    
        self.data_pipeline = None
        self.data_cleaner_params = data_cleaner_params
        self.data_cleaner = None
        self.knn_model = None
        self.fit_kwargs = None
        self.class_balancing = None
        self.classes_ = None
        self._build_model(space_sample)

        return 

    def _build_model(self, space_sample):
        """This function builds a kNN model, space_sample should be a search space, 
        the compile_and_forward, which is used to..., comes from search_space.
        """
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
        pipeline_module = space.get_inputs(outputs[0])
        assert len(pipeline_module) == 1, 'The `HyperEstimator` can only contains 1 input.'
        assert isinstance(pipeline_module[0],
                          ComposeTransformer), 'The upstream node of `HyperEstimator` must be `ComposeTransformer`.'
        self.data_pipeline = self.build_pipeline(space, pipeline_module[0])

        if self.data_cleaner_params is not None:
            self.data_cleaner = DataCleaner(**self.data_cleaner_params)
        else:
            self.data_cleaner = None

    def build_pipeline(self, space, last_transformer):
        transformers = []
        while True:
            next, (name, p) = last_transformer.compose()
            transformers.insert(0, (name, p))
            inputs = space.get_inputs(next)
            if inputs == space.get_inputs():
                break
            assert len(inputs) == 1, 'The `ComposeTransformer` can only contains 1 input.'
            assert isinstance(inputs[0],
                              ComposeTransformer), 'The upstream node of `ComposeTransformer` must be `ComposeTransformer`.'
            last_transformer = inputs[0]
        assert len(transformers) > 0
        if len(transformers) == 1:
            return transformers[0][1]
        else:
            pipeline = sk_pipeline.Pipeline(steps=transformers)
            return pipeline

    @cache(arg_keys='X,y', attr_keys='data_cleaner_params,pipeline_signature',
           attrs_to_restore='data_cleaner,data_pipeline',
           transformer='transform_data')
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