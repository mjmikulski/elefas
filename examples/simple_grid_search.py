from elefas.optimizations import GridSearch
from elefas.hyperparameters import Choice, Linear, Geometric

space = GridSearch()


space.add(Choice('optimizer', 'adam'))
space.add(Choice('activation', ['tanh', 'sigmoid', 'relu']))
space.add(Linear('dense_dropout', 0.0, 0.6, n=7))
space.add(Linear('conv_dropout', b=0.4, step=0.05))
space.add(Geometric('dense_size', 8, 256, step=2))
# space.add(Fibonacci('batch_size', m=4, n=5))

space.compile()

z = 0
for p in space():
    z += 1

print(f'z: {z}')




print('Done.')