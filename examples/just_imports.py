from elefas.optimizations import RandomSearch
from elefas.hyperparameters import Choice

r = RandomSearch()
r.add(Choice('dropout', 0.5))

print('Done.')