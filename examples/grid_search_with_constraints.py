from elefas.hyperparameters import Boolean, Linear, Exponential, Constraint
from elefas.spaces import Grid

space = Grid()

space.add(Exponential('learning_rate', 1.0E-5, 0.1))
space.add(Linear('momentum', 0, 0.9), n=3)
space.add(Boolean('Nesterov'))
space.add(Exponential('batch_size', start=10, stop=1000), step=4)

# no point for Nesterov if no momentum
space.add(Constraint('pointless_nesterov', f=lambda momentum, Nesterov: not (Nesterov and momentum == 0)))

# skip models that we think will converge to slowly
# names of lambda parameters are just the names of hyper-parameters that we already added
space.add(Constraint('learning_speed', f=lambda learning_rate, batch_size: learning_rate/batch_size > 1.E-6))


space.compile()

for p in space:
    print(p)


space.summary()