from hypertoy.pipeline import DataFrameMapper
from hypertoy.estimator import ComplexKnn
from .sklearn_ import cat_pipeline_simple

from hypernets.core.ops import ModuleChoice, HyperInput
from hypernets.core.search_space import Choice, Real, Int, Bool
from hypernets.core.search_space import HyperSpace

def search_space():
    space = HyperSpace()
    with space.as_default():
        hyper_input = HyperInput(name='input1')
        cat_pipeline = cat_pipeline_simple()(hyper_input)
    

        knn_params = {'n_neighbors': Choice([1, 3, 5]),
                'weights': Choice(['uniform', 'distance']),
                'algorithm': Choice(['auto', 'ball_tree', 'kd_tree', 'brute']),
                'leaf_size': Choice([10, 20 ,30]),
                'p': Choice([1, 2]),
                'metric': 'minkowski',
                'metric_params': None,
                'n_jobs': None
        }

        knn_est = ComplexKnn(fit_kwargs={}, **knn_params)
        knn_est(cat_pipeline)
        space.set_inputs(hyper_input)    
    return space