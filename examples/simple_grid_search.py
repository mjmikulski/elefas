from elefas.optimizations import GridSearch
from elefas.hyperparameters import Choice, Linear, Exponential

space = GridSearch()


# space.add(Constant('optimizer', 'adam'))

space.add(Choice('activation', ['tanh', 'sigmoid', 'relu']))

space.add(Linear('dense_dropout', 0.0, 0.6), n=7)
space.add(Linear('conv_dropout', 0.0, 0.4), step=0.05)

space.add(Exponential('batch_size', 10, 1000))  # default n = 5 will be used
space.add(Exponential('dense_size', 8, 256), step=2)


space.compile()

for p in space():
    print(p)


space.summary()


print('Done.')