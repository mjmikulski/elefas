from elefas.hyperparameters import Linear, Choice, Dependent
from elefas.optimizations import Grid

space = Grid()

# Add a few hyper-parameters that take the same values, but are independent.
space.add(Choice(['dense01_activation', 'dense02_activation'], ['tanh', 'sigmoid', 'relu']))
space.add(Linear(['dense01_dropout', 'dense02_dropout'], 0., 0.6), n=4)

# Add dependent hyper-parameters
space.add(Choice('dense01_dim', [64,128]))
space.add(Dependent('dense02_dim', f=lambda dense01_dim: dense01_dim//2 ))


space.compile()

for p in space():
    print(p)

space.summary()

