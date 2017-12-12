from elefas.hyperparameters import Linear, Choice, Dependent
from elefas.optimizations import GridSearch

space = GridSearch()


space.add(Choice(['dense01_activation', 'dense02_activation'], ['tanh', 'sigmoid', 'relu']))
space.add(Linear(['dense01_dropout', 'dense02_dropout'], 0., 0.6), n=4)




space.add(Choice('dense01_dim', [64,128]))
space.add(Dependent('dense02_dim', f=lambda dense01_dim: dense01_dim//2 ))


space.compile()

for p in space():
    print(p)


space.summary()