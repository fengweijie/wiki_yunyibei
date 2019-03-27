from keras.layers import Dense, Input, Flatten

from keras.layers import GlobalMaxPool1D, Bidirectional, Convolution1D, Embedding, BatchNormalization, MaxPooling1D, \
    Dropout, LSTM


from keras import backend as K
'''
# backend 通常用来获得中间层的输出
from keras import backend as K
# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([X])[0]

注意，如果你的模型在训练和测试两种模式下不完全一致，
例如你的模型中含有Dropout层，批规范化（BatchNormalization）层等组件，
你需要在函数中传递一个learning_phase的标记，像这样：
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])
# output in test mode = 0
layer_output = get_3rd_layer_output([X, 0])[0]

# output in train mode = 1
layer_output = get_3rd_layer_output([X, 1])[0]

'''


from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints

import numpy as np
from keras.models import Model

from keras.layers.merge import Concatenate



from keras.callbacks import EarlyStopping, ModelCheckpoint
'''
可以定义EarlyStopping来提前终止训练

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])

'''

import pandas as pd

r = np.load("./cache/data.npz")  # 加载一次即可
x_train = r['arr_0']
y_train = r['arr_1']
x_val = r['arr_2']
y_val = r['arr_3']
X_te = r['arr_4']
embedding_matrix = r['arr_5']

max_features = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128

num_lstm = 150
rate_drop_lstm = 0.25
rate_drop_dense = 0.25
# 在最优方案5的基础上进行修改阈值4.7，分别将其改为4.6,4.7,4.8没有对其进行修改，之后还试验了
# final_score[final_score>4.7] =5发现等于4.7的结果最优

embedding_layer = Embedding(max_features,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True)


# 双向LSTM模型
def get_lstm_model():
    inp = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(inp)
    x = Bidirectional(LSTM(250, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True))(
        embedded_sequences)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="linear")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='mse',
                  optimizer='adam')
    return model


# attention model
num_lstm = 300
num_dense = 256
rate_drop_lstm = 0.25
rate_drop_dense = 0.25

act = 'relu'


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim


def get_attention_model():
    inp = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(inp)
    x = lstm_layer(embedded_sequences)
    x = Dropout(rate_drop_dense)(x)
    merged = Attention(MAX_SEQUENCE_LENGTH)(x)
    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    preds = Dense(1, activation='linear')(merged)
    model = Model(inputs=inp, outputs=preds)
    model.compile(loss='mse',
                  optimizer='adam')

    return model


# CNN模型

filter_sizes = (2, 3, 4, 5)
num_filters = 10
dropout_prob = (0.1, 0.1)


def get_CNN_model():
    inp = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    x = embedding_layer(inp)
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(x)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    x = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    x = BatchNormalization()(x)
    x = Dense(50, activation="relu")(x)
    x = Dense(1, activation="linear")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='mse',
                  optimizer='adam')

    return model


def get_model(model_name):
    if model_name == 'RNN':
        return get_lstm_model()
    elif model_name == 'Attention':
        return get_attention_model()
    elif model_name == 'CNN':
        return get_CNN_model()


def predict(model_name):
    model = get_model(model_name)
    batch_size = 128
    epochs = 100

    file_path = "./cache/weights_base.best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

    callbacks_list = [checkpoint, early]
    model.fit(x_train, y_train, batch_size=batch_size, epochs=100, validation_data=(x_val, y_val),
              callbacks=callbacks_list)

    model.load_weights(file_path)

    y_test = model.predict(X_te)

    y_test[y_test > 5] = 5

    test = pd.read_csv('./cache/predict-processed.csv')
    sub = pd.DataFrame()
    sub['id'] = pd.DataFrame(test["Id"])
    sub['Score'] = pd.DataFrame(y_test)
    sub.to_csv('./cache/sub_{}.csv'.format(model_name), index=False, header=False)


predict("Attention")
predict("RNN")
predict("CNN")


