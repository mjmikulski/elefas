from elefas.optimizations import GridSearch
from elefas.hyperparameters import Choice, Linear, Geometric


space = RandomSearch()

space.add(Exponential('learning_rate', 1.0E-5, 0.1))
space.add(Linear('momentum', 0, 0.95))
space.add(Boolean('Nesterov'))
space.add(Exponential('batch_size', a=10, b=1000))

space.add(Constraint(['learning_rate', 'batch_size'], f=lambda lr, bs: lr/bs > 1.E-6))


space.compile()

for p in space():
    print(p)



print('Done.')