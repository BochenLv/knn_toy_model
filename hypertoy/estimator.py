from sklearn import neighbors

from hypernets.core.search_space import ModuleSpace

class HyperEstimator(ModuleSpace):
    def __init__(self, fit_kwargs, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)
        self.fit_kwargs = fit_kwargs
        self.estimator = None
        self.class_balancing = False

    def _build_estimator(self, task, kwargs):
        raise NotImplementedError

    def build_estimator(self, task):
        pv = self.param_values
        if pv.__contains__('class_balancing'):
            self.class_balancing = pv.pop('class_balancing')
        self.estimator = self._build_estimator(task, pv)
    
    def _compile(self):
        #?
        pass

    def _forward(self, inputs):
        return self.estimator

    def _on_params_ready(self):
        pass

class KNNClassifierWrapper(neighbors.KNeighborsClassifier):
    def fit(self, X, y, **kwargs):
        task = self.__dict__.get('task')
        super(KNNClassifierWrapper, self).fit(X, y, **kwargs)

    def predict_proba(self, X, **kwargs):
        proba = super(KNNClassifierWrapper, self).predict_proba(X, **kwargs)
        return proba

    @property
    def iteration_scores(self):
        scores = []
        if self.evals_result_:
            valid = self.evals_result_.get('valid_0')
            if valid:
                scores = list(valid.values())[0] 
        return scores

class KNNRegressorWrapper(neighbors.KNeighborsRegressor):
    def fit(self, X, y, **kwargs):
        super(KNNRegressorWrapper, self).fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        pred = super(KNNRegressorWrapper, self).predict(X, **kwargs)
        return pred

    @property
    def iteration_scores(self):
        scores = []
        if self.evals_result_:
            valid = self.evals_result_.get('valid_0')
            if valid:
                scores = list(valid.values())[0]
        return scores

class kNNEstimator(HyperEstimator):
    def __init__(self, fit_kwargs, n_neighbors=2, weights='uniform', algorithm='brute', 
                    leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None,
                     space=None, name=None, **kwargs):
        if n_neighbors is not None and n_neighbors != 2:
            kwargs['n_neighbors'] = n_neighbors
        if weights is not None and weights != 'uniform':
            kwargs['weights'] = weights
        if algorithm is not None and algorithm != 'brute':
            kwargs['algorithm'] = algorithm
        if leaf_size is not None and leaf_size != 30:
            kwargs['leaf_size'] = leaf_size
        if p is not None and p != 2:
            kwargs['p'] = p
        if metric is not None and metric != 'minkowski':
            kwargs['metric'] = metric
        if metric_params is not None:
            kwargs['metric_params'] = metric_params
        if n_jobs is not None:
            kwargs['n_jobs'] = n_jobs
        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, kwargs):
        if task == 'regression':
            knn = KNNRegressorWrapper(**kwargs)
        else:
            knn = KNNClassifierWrapper(**kwargs)
        return knn


