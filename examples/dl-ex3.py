import keras
from datetime import timedelta
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD

from elefas.engine import normalize
from elefas.hyperparameters import Exponential, Linear, Dependent, Constraint, Constant
from elefas.optimizations import Random, Grid

# load and prepare data
num_classes = 10
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# limit data
(x_train, y_train) = (x_train[:6000], y_train[:6000])
(x_test, y_test) = (x_test[:1000], y_test[:1000])
# on full data it arrives at 91.9% accuracy, 110 sec/epoch on Intel i7-7700K
# best hyper-parameters set found by this procedure is
# [('batch_size', 324), ('momentum', 0.9666666666666667), ('lr', 0.027)]

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


def build_and_fit_model(space):
    space.compile()
    for p in space:
        space.status()  # show current point

        # build model using hyper-parameters
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        opt = SGD(lr=p['lr'], momentum=p['momentum'])

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=p['batch_size'],
                  epochs=12,
                  verbose=2,
                  validation_data=(x_test, y_test))

        # evaluate model
        loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
        print(f'Test loss: {loss:.2f}')
        print(f'Test accuracy: {accuracy:.3f}')
        space.add_score(val_loss=loss, val_accu=accuracy)




# phase 1: batch_size
s1 = Grid()

s1.add(Exponential('batch_size', 4, 256), step=2)
s1.add(Constant('lr', 0.001))
s1.add(Constant('momentum', 0.9))

build_and_fit_model(s1)

best_val_accu, best_point = s1.best_sp('val_accu')
print(f'Best validation accuracy {best_val_accu:.3f} for {best_point}\n')
s1.summary()


# phase 2: lr
best_batch_size = best_point['batch_size']

s2 = Grid()  # in future: Monotonic()
s2.add(Exponential('batch_size', best_batch_size, 2000), step=3)
s2.add(Dependent('lr', f=lambda batch_size: 0.001 * batch_size / best_batch_size))
s2.add(Constant('momentum', 0.9))

build_and_fit_model(s2)

best_val_accu, best_point = s2.best_sp('val_accu')
print(f'Best validation accuracy {best_val_accu:.3f} for {best_point}\n')
s2.summary()


# phase 3: momentum
best_lr = best_point['lr']
best_batch_size = best_point['batch_size']

s3 = Grid()
s3.add(Exponential('batch_size', best_batch_size, 6000), step=3)
s3.add(Constant('lr', best_lr))
s3.add(Dependent('momentum', f=lambda batch_size: 1 - 0.1 * best_batch_size / batch_size))

build_and_fit_model(s3)

best_val_accu, best_point = s3.best_sp('val_accu')
print(f'Best validation accuracy {best_val_accu:.3f} for {best_point}\n')
s3.summary()


# phase 4: fine batch_size
# to be done