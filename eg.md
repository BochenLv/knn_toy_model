```python
def cnn_search_space(input_shape, output_units, block_num_choices=[2, 3], kernel_size_choices=[(1, 1), (3, 3)]):
        input = Input(shape=input_shape)
        blocks = Repeat(
            lambda step: conv_block(
                block_no=step,
                hp_pooling=Choice([0, 1]),
                hp_filters=Choice([32, 64]]),
                hp_kernel_size=Choice(kernel_size_choices),
                hp_use_bn=Bool(),
                hp_activation='relu',
                hp_bn_act=Choice([seq for seq in itertools.permutations(range(2))])),
            repeat_times=block_num_choices)(input)
        x = Flatten()(blocks)
        x = Dense(units=hp_fc_units, activation=hp_activation, name='fc1')(x)
        x = Dense(output_units, activation=output_activation, name='predictions')(x)
    return space
```
```python
search_space = search_space_general
```

eg2
```python
searcher = MCTSSearcher(search_space, max_node_space=10)
```

```python
model = HyperGBM(searcher, task='multiclass', reward_metric='accuracy')
model.search(X_train, y_train, X_eval=X_test, y_eval=y_test)
```

```python
best_model = model.load_estimator(model.get_best_trial().model_file)
```