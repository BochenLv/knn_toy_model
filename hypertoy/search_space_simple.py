import numpy as np

from hypertoy.estimator import ComplexKnn
from hypertoy.pipeline import Pipeline

from hypernets.core.ops import HyperInput
from hypernets.core.search_space import Choice
from hypernets.core.search_space import HyperSpace
from hypernets.tabular import column_selector

from sklearn_.transformers import SimpleImputer, SafeOrdinalEncoder


def search_space():
    space = HyperSpace()
    with space.as_default():
        hyper_input = HyperInput(name='input1')
        
        # build the categorical pipeline
        cs = column_selector.column_object_category_bool
        cat_pipeline = Pipeline([
        SimpleImputer(missing_values=np.nan, strategy='constant', name=f'categorical_imputer_{0}'),
        SafeOrdinalEncoder(name=f'categorical_label_encoder_{0}', dtype='int32')],
        columns=cs,
        name=f'categorical_pipeline_simple_{0}',
        )(hyper_input)

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