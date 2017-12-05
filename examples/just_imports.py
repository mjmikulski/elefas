from elefas.optimizations import RandomSearch
from elefas.hyperparameters import Constant

r = RandomSearch()
r.add(Constant())

print('Done.')