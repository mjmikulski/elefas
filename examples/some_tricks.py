from time import sleep

from elefas.hyperparameters import Linear, Choice, Dependent, Constraint
from elefas.optimizations import Grid, Random

space = Random(time_limit=15)

# Add a few hyper-parameters that take the same values, but are independent.
space.add(Choice(['dense01_activation', 'dense02_activation'], ['tanh', 'sigmoid', 'relu']))
space.add(Linear(['dense01_dropout', 'dense02_dropout'], 0., 0.6))

# Add dependent hyper-parameters
space.add(Choice('dense01_dim', [64,128]))
space.add(Dependent('dense02_dim', f=lambda dense01_dim: dense01_dim//2 ))

space.add(Constraint('limited_dropout', f=lambda dense01_dropout, dense02_dim: dense01_dropout < 0.5 or dense02_dim > 50 ))

space.compile()

for p in space:
    print(p)
    sleep(0.5)

space.summary()

