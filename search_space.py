
from .estimator import kNNEstimator
from .pipeline import DataFrameMapper
from .cfg import KnnCfg as cfg


from .sklearn.sklearn_ops import numeric_pipeline_simple, numeric_pipeline_complex, \
    categorical_pipeline_simple, categorical_pipeline_complex
from Hypernets.hypernets.core.ops import ModuleChoice, HyperInput
from Hypernets.hypernets.core.search_space import HyperSpace
from Hypernets.hypernets.core.search_space import Choice
from Hypernets.hypernets.tabular.column_selector import column_object




def _merge_dict(*args):
    d = {}
    for a in args:
        if isinstance(a, dict):
            d.update(a)
    return d


class _HyperEstimatorCreator(object):
    def __init__(self, cls, init_kwargs, fit_kwargs):
        super(_HyperEstimatorCreator, self).__init__()

        self.estimator_cls = cls
        self.estimator_fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
        self.estimator_init_kwargs = init_kwargs if init_kwargs is not None else {}

    def __call__(self, *args, **kwargs):
        return self.estimator_cls(self.estimator_fit_kwargs, **self.estimator_init_kwargs)


class SearchSpaceGenerator(object):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        self.options = kwargs

    @property
    def estimators(self):
        raise NotImplementedError()

    def create_preprocessor(self, hyper_input, options):
        cat_pipeline_mode = options.pop('cat_pipeline_mode', cfg.category_pipeline_mode)
        num_pipeline_mode = options.pop('num_pipeline_mode', cfg.numeric_pipeline_mode)
        dataframe_mapper_default = options.pop('dataframe_mapper_default', False)

        pipelines = []
        if cfg.category_pipeline_enabled:
            if cat_pipeline_mode == 'simple':
                pipelines.append(categorical_pipeline_simple()(hyper_input))
            else:
                pipelines.append(categorical_pipeline_complex()(hyper_input))

        if num_pipeline_mode == 'simple':
            pipelines.append(numeric_pipeline_simple()(hyper_input))
        else:
            pipelines.append(numeric_pipeline_complex()(hyper_input))

        preprocessor = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                       df_out_dtype_transforms=[(column_object, 'int')])(pipelines)

        return preprocessor

    def create_estimators(self, hyper_input, options):
        assert len(self.estimators.keys()) > 0

        creators = [_HyperEstimatorCreator(pairs[0],
                                           init_kwargs=_merge_dict(pairs[1], options.pop(f'{k}_init_kwargs', None)),
                                           fit_kwargs=_merge_dict(pairs[2], options.pop(f'{k}_fit_kwargs', None)))
                    for k, pairs in self.estimators.items()]

        estimators = [c() for c in creators]
        return ModuleChoice(estimators, name='estimator_options')(hyper_input)

    def __call__(self, *args, **kwargs):
        options = _merge_dict(self.options, kwargs)

        space = HyperSpace()
        with space.as_default():
            hyper_input = HyperInput(name='input1')
            self.create_estimators(self.create_preprocessor(hyper_input, options), options)
            space.set_inputs(hyper_input)
        return space


class KNNSearchSpaceGenerator(SearchSpaceGenerator):
    def __init__(self, **kwargs):
        super(KNNSearchSpaceGenerator, self).__init__(**kwargs)

    @property
    def default_knn_init_kwargs(self):
        return {'n_neighbors': Choice([1, 3, 5]),
                'weights': Choice(['uniform', 'distance']),
                'algorithm': Choice(['auto', 'ball_tree', 'kd_tree', 'brute']),
                'leaf_size': Choice([10, 20 ,30]),
                'p': Choice([1, 2]),
                'metric': 'minkowski',
                'metric_params': None,
                'n_jobs': None}
    
    @property
    def default_knn_fit_kwargs(self):
        return {}

    def estimators(self):
        r = {'knn': (kNNEstimator, self.default_knn_init_kwargs, self.default_knn_fit_kwargs)}
        return r

search_space_eg = KNNSearchSpaceGenerator()