```python
def get_space():
    space = HyperSpace()
    with space.as_default():
        in1 = Input(shape=(10,))
        in2 = Input(shape=(20,))
        in3 = Input(shape=(1,))
        concat = Concatenate()([in1, in2, in3])
        dense1 = Dense(10, activation=Choice(['relu', 'tanh', None]), use_bias=Boll())(concat)
        bn1 = BatchNormalization()(dense1)
        dropout1 = Dropout(Choice([0.3, 0.4, 0.5]))(bn1)
        output = Dense(2, activation='softmax', use_bias=True)(dropout1)
    return space
```