# -*- coding:utf-8 -*-
"""
This file is the same as the orginal file in the HyperGBM, since classes and functions in
this file are actuall not designed specifically for GBMs thus can be used broadlly for many
differen models(estimators). These functions and classes are in fact dealing with the data 
preprocessing and assembling them into the full-pipeline, which appear in the search space
before the data being sent to estimators.
"""
import numpy as np
from hypertoy.search_space import sear_space_eg

from hypertoy.cfg import KnnCfg as cfg
from hypertoy.pipeline import Pipeline
from sklearn.transformers import SimpleImputer, SafeOneHotEncoder, TruncatedSVD, \
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, SafeOrdinalEncoder, \
    LogStandardScaler, DatetimeEncoder, TfidfEncoder

from hypernets.core.ops import ModuleChoice, Optional, Choice
from hypernets.tabular import column_selector


def categorical_pipeline_simple(impute_strategy='constant', seq_no=0):
    if cfg.category_pipeline_auto_detect:
        cs = column_selector.AutoCategoryColumnSelector(
            dtype_include=column_selector.column_object_category_bool.dtype_include,
            cat_exponent=cfg.category_pipeline_auto_detect_exponent)
    else:
        cs = column_selector.column_object_category_bool
    pipeline = Pipeline([
        SimpleImputer(missing_values=np.nan, strategy=impute_strategy, name=f'categorical_imputer_{seq_no}'),
        SafeOrdinalEncoder(name=f'categorical_label_encoder_{seq_no}', dtype='int32')
        # MultiLabelEncoder(name=f'categorical_label_encoder_{seq_no}')
    ],
        columns=cs,
        name=f'categorical_pipeline_simple_{seq_no}',
    )
    return pipeline


def categorical_pipeline_complex(impute_strategy=None, svd_components=3, seq_no=0):
    if impute_strategy is None:
        impute_strategy = Choice(['constant', 'most_frequent'])
    elif isinstance(impute_strategy, list):
        impute_strategy = Choice(impute_strategy)
    if isinstance(svd_components, list):
        svd_components = Choice(svd_components)

    def onehot_svd():
        onehot = SafeOneHotEncoder(name=f'categorical_onehot_{seq_no}')
        optional_svd = Optional(TruncatedSVD(n_components=svd_components, name=f'categorical_svd_{seq_no}'),
                                name=f'categorical_optional_svd_{seq_no}',
                                keep_link=True)(onehot)
        return optional_svd

    imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strategy, name=f'categorical_imputer_{seq_no}')
    label_encoder = SafeOrdinalEncoder(name=f'categorical_label_encoder_{seq_no}')
    onehot = onehot_svd()
    le_or_onehot_pca = ModuleChoice([label_encoder, onehot], name=f'categorical_le_or_onehot_pca_{seq_no}')

    if cfg.category_pipeline_auto_detect:
        cs = column_selector.AutoCategoryColumnSelector(
            dtype_include=column_selector.column_object_category_bool.dtype_include,
            cat_exponent=cfg.category_pipeline_auto_detect_exponent)
    else:
        cs = column_selector.column_object_category_bool
    pipeline = Pipeline([imputer, le_or_onehot_pca],
                        name=f'categorical_pipeline_complex_{seq_no}',
                        columns=cs)
    return pipeline


def numeric_pipeline_simple(impute_strategy='mean', seq_no=0):
    pipeline = Pipeline([
        SimpleImputer(missing_values=np.nan, strategy=impute_strategy,
                      name=f'numeric_imputer_{seq_no}', force_output_as_float=True),
        StandardScaler(name=f'numeric_standard_scaler_{seq_no}')
    ],
        columns=column_selector.column_number_exclude_timedelta,
        name=f'numeric_pipeline_simple_{seq_no}',
    )
    return pipeline


def numeric_pipeline_complex(impute_strategy=None, seq_no=0):
    if impute_strategy is None:
        impute_strategy = Choice(['mean', 'median', 'constant', 'most_frequent'])
    elif isinstance(impute_strategy, list):
        impute_strategy = Choice(impute_strategy)

    imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strategy, name=f'numeric_imputer_{seq_no}',
                            force_output_as_float=True)
    scaler_options = ModuleChoice(
        [
            LogStandardScaler(name=f'numeric_log_standard_scaler_{seq_no}'),
            StandardScaler(name=f'numeric_standard_scaler_{seq_no}'),
            MinMaxScaler(name=f'numeric_minmax_scaler_{seq_no}'),
            MaxAbsScaler(name=f'numeric_maxabs_scaler_{seq_no}'),
            RobustScaler(name=f'numeric_robust_scaler_{seq_no}')
        ], name=f'numeric_or_scaler_{seq_no}'
    )
    scaler_optional = Optional(scaler_options, keep_link=True, name=f'numeric_scaler_optional_{seq_no}')
    pipeline = Pipeline([imputer, scaler_optional],
                        name=f'numeric_pipeline_complex_{seq_no}',
                        columns=column_selector.column_number_exclude_timedelta)
    return pipeline


def datetime_pipeline_simple(impute_strategy='constant', seq_no=0):
    pipeline = Pipeline([
        DatetimeEncoder(name=f'datetime_encoder_{seq_no}'),
        SimpleImputer(missing_values=np.nan, strategy=impute_strategy, fill_value=0,
                      name=f'datetime_imputer_{seq_no}'),
    ],
        columns=column_selector.column_all_datetime,
        name=f'datetime_pipeline_simple_{seq_no}',
    )
    return pipeline


def text_pipeline_simple(impute_strategy='constant', svd_components=3, seq_no=0):
    if isinstance(impute_strategy, list):
        impute_strategy = Choice(impute_strategy)
    if isinstance(svd_components, list):
        svd_components = Choice(svd_components)

    cs = column_selector.TextColumnSelector(dtype_include=column_selector.column_text.dtype_include,
                                            word_count_threshold=cfg.text_pipeline_word_count_threshold)
    pipeline = Pipeline([
        SimpleImputer(missing_values=np.nan, strategy=impute_strategy, name=f'text_imputer_{seq_no}'),
        TfidfEncoder(name='text_tfidf_{seq_no}'),
        TruncatedSVD(n_components=svd_components, name=f'text_svd_{seq_no}'),
    ],
        columns=cs,
        name=f'a_text_pipeline_simple_{seq_no}',
    )
    return pipeline
