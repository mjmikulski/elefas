from elefas.hyperparameters import Choice, Linear, Exponential
from elefas.optimizations import Random

space = Random(20)

space.add(Choice('initialization', ['zero', 'lecun_uniform', 'glorot_normal', 'he_normal']))

space.add(Exponential('fc_size', 16, 32))
space.add(Linear('dropout', 0.0, 0.6))

space.add(Exponential('lr', 0.0001, 0.1))
space.add(Linear('batch_size', 10, 100))

space.compile()

for p in space:
    print(p)

space.summary()