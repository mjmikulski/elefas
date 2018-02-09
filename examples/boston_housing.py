import math

from keras.datasets import boston_housing
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD

from elefas.engine import normalize
from elefas.hyperparameters import Linear, Exponential, Choice, Boolean, Constraint
from elefas.optimizations import Random

# prepare data:
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
x_train, x_test = normalize(x_train, x_test)

print(f'x_train.shape: {x_train.shape}')
print(f'x_test.shape: {x_test.shape}')

# define hyper-parameters
space = Random(100)

space.add(Exponential('dense_1_units', 10, 100))
space.add(Exponential('dense_2_units', 5, 50))
space.add(Constraint('dense size should decrease', f=lambda dense_1_units, dense_2_units: dense_1_units > dense_2_units))
space.add(Choice(['activation_1', 'activation_2'], ['tanh', 'relu', 'sigmoid']))
space.add(Linear('dropout', 0, 0.8))

space.add(Exponential('lr', 1e-6, 1))
space.add(Linear('momentum', 0, 0.999))
space.add(Boolean('nesterov'))

space.add(Exponential('batch_size', 8, 128))
space.add(Linear('epochs', 10, 500))

space.compile()

best_loss = math.inf
best_p = None

for p in space():
    print('Exploring: ', p)

    model = Sequential()
    model.add(Dense(units=p['dense_1_units'], input_dim=x_train.shape[1]))
    model.add(Activation(p['activation_1']))
    model.add(Dropout(p['dropout']))

    model.add(Dense(units=p['dense_2_units']))
    model.add(Activation(p['activation_2']))
    model.add(Dropout(p['dropout']))

    model.add(Dense(1))

    opt = SGD(lr=p['lr'], momentum=p['momentum'], nesterov=p['nesterov'])

    model.compile(optimizer=opt, loss='mse')

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

print(f'Best loss so far is {best_loss:.2f} for {best_p}')
space.summary()
