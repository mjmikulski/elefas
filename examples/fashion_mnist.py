import keras
from datetime import timedelta
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from elefas.engine import normalize
from elefas.hyperparameters import Exponential, Linear, Dependent, Constraint
from elefas.optimizations import Random


# load and prepare data
num_classes = 10
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = normalize(x_train, x_test)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# add hyper-parameters
space = Random(time_limit=timedelta(hours=6))

space.add(Linear('conv_dropout', 0, 0.3))
space.add(Dependent('dense_dropout', f=lambda conv_dropout: 2 * conv_dropout))

space.add(Exponential('batch_size', 4, 256))
space.add(Exponential('lr', 1e-4, 1e-2))
space.add(Linear('epochs', 5, 20))

space.add(Constraint('skip slow convergence', f=lambda lr, batch_size: lr/batch_size > 1e-5 ))

space.compile()

best_p = None
best_accuracy = 0

for p in space:
    space.status()  # show current point

    # build model using hyper-parameters
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(p['conv_dropout']))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(p['dense_dropout']))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=p['lr']),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=p['batch_size'],
              epochs=p['epochs'],
              verbose=2,
              validation_data=(x_test, y_test))

    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_p = p
        print('This is new best accuracy')
    else:
        print(f'Best accuracy so far is {best_accuracy:.2f}')

    print('')

print(f'Best is {best_accuracy:.2f} for {best_p}')

space.summary()
