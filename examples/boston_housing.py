import math
import random

from keras.datasets import boston_housing
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD

from elefas.engine import normalize
from elefas.hyperparameters import Linear, Exponential, Choice, Boolean, Constraint
from elefas.optimizations import Random


# fix seed
random.seed(11)

# prepare data:
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
x_train, x_test = normalize(x_train, x_test)

print(f'x_train.shape: {x_train.shape}')
print(f'x_test.shape: {x_test.shape}')

# define hyper-parameters
from datetime import timedelta
space = Random(points=100, time_limit=(timedelta(minutes=15)))

space.add(Exponential('dense_1_units', 20, 100))
space.add(Exponential('dense_2_units', 10, 50))
space.add(Exponential('dense_3_units', 5, 25))
space.add(Constraint('dense size should decrease', f=lambda dense_1_units, dense_2_units, dense_3_units: dense_1_units > dense_2_units > dense_3_units))
space.add(Choice(['activation_1', 'activation_2', 'activation_3'], ['tanh', 'relu', 'sigmoid']))
space.add(Linear(['dropout_1', 'dropout_2', 'dropout_3'], 0, 0.6))
space.add(Boolean(['BN_1', 'BN_2', 'BN_3']))

space.add(Exponential('lr', 1e-5, 1e-1))
space.add(Linear('momentum', 0.5, 0.999))
space.add(Boolean('nesterov'))

space.add(Exponential('batch_size', 8, 64))
space.add(Linear('epochs', 300, 1200))

space.compile()

best_loss = math.inf
best_p = None

for p in space:
    space.status()

    model = Sequential()

    model.add(Dense(units=p['dense_1_units'], input_dim=x_train.shape[1]))
    if p['BN_1']:
        model.add(BatchNormalization())
    model.add(Activation(p['activation_1']))
    model.add(Dropout(p['dropout_1']))

    model.add(Dense(units=p['dense_2_units']))
    if p['BN_2']:
        model.add(BatchNormalization())
    model.add(Activation(p['activation_2']))
    model.add(Dropout(p['dropout_2']))

    model.add(Dense(units=p['dense_3_units']))
    if p['BN_3']:
        model.add(BatchNormalization())
    model.add(Activation(p['activation_3']))
    model.add(Dropout(p['dropout_3']))

    model.add(Dense(1))

    opt = SGD(lr=p['lr'], momentum=p['momentum'], nesterov=p['nesterov'])

    model.compile(optimizer=opt, loss='mape')

    model.summary()

    model.fit(x_train, y_train,
              batch_size=p['batch_size'],
              epochs=p['epochs'],
              validation_data=(x_test, y_test),
              shuffle=True,
              verbose=0)

    # Score trained model.
    loss = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test loss: {loss:.2f}')

    if loss < best_loss:
        best_loss = loss
        best_p = p
        print('This is new best loss')
    else:
        print(f'Best loss so far is {best_loss:.2f}')

    print('')

print(f'Best is {best_loss:.2f} for {best_p}')
space.summary()
