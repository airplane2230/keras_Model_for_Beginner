import tensorflow as tf

from tensorflow.keras import layers

def create_inputs():
    inputs = {}
    
    # Numeric, categorical feature를 나눠놓았을 때 경우를 다룸
    for col in NUMERIC_COLS:
        inputs[col] = Input(name = col, shape = (1, ), dtype = tf.float32)

    for col in CATEGORICAL_COLS:
        inputs[col] = Input(name = col, shape = (1, ))

    return inputs

def encode_inputs(inputs, encoding_size):
    encoded_features = []

    for feature_name in inputs:
        if feature_name in CATEGORICAL_COLS:
            emb_input_len = int(max(datas[col].unique())) + 1
            encoded_feature = tf.keras.layers.Embedding(input_dim = emb_input_len,
                                                        output_dim = encoding_size,
                                                        name = feature_name + '_embedding')(inputs[feature_name])
            encoded_feature = layers.Flatten()(encoded_feature)
        else:
            # encoded_feature = tf.expand_dims(inputs[feature_name], -1)
            encoded_feature = tf.keras.layers.Dense(encoding_size, activation = 'relu')(inputs[feature_name])
            
            encoded_features.append(encoded_feature)

    return encoded_features

class GatedLinearUnit(layers.Layer):
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.linear = layers.Dense(units)
        self.sigmoid = layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)
    
class GatedResidualNetwork(layers.Layer):
    def __init__(self, units, dropout_rate):
        super(GatedResidualNetwork, self).__init__()
        self.units = units
        self.elu_dense = layers.Dense(units, activation="relu")
        self.linear_dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        # tf 2.3 version 이상
        self.layer_norm = layers.LayerNormalization()
        self.project = layers.Dense(units)

    def call(self, inputs):
        # 1.
        x = self.elu_dense(inputs)
        # 2.
        x = self.linear_dense(x)
        x = self.dropout(x)
        # 3.
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        # 4.
        x = self.layer_norm(x)
        
        return x
    
def get_model(num_features, encoding_size):
    inputs = create_inputs()
    features = encode_inputs(inputs, encoding_size)

    grns = list()
    for idx in range(num_features):
        grn = GatedResidualNetwork(encoding_size, dropout_rate = 0.1)
        grns.append(grn)

    v = layers.concatenate(features)
    v = GatedResidualNetwork(encoding_size, dropout_rate = 0.1)(v)
    softmax_layer = layers.Dense(units = num_features, activation = 'softmax')
    v = tf.expand_dims(softmax_layer(v), axis = -1)

    x = []
    for idx, feature in enumerate(features):
        x.append(grns[idx](feature))

    x = tf.stack(x, axis = 1)

    outputs = tf.squeeze(tf.matmul(v, x, transpose_a=True), axis = 1)
    outputs = layers.Dense(1, activation = 'linear')(outputs)

    model = tf.keras.models.Model(inputs = inputs, outputs = outputs)

    return model

'''
    refer : https://keras.io/examples/structured_data/classification_with_grn_and_vsn/
'''

num_features = datas.shape[-1]
encoding_size = 32

model = get_model(num_features, encoding_size)
# model.compile(optimizer = 'adam', loss = 'mae')
# model.summary()

'''

    Data Feeding Example is below
    
    NUMERIC_COLS = [
                'A', 'B', 'C', 'D', 'E',
]

    CATEGORICAL_COLS = [
            'F', 'G', 'H', 'I'
]

    # datas -> pd.DataFrame, target -> pd.Series
    for col in NUMERIC_COLS:
        datas[col] = (datas[col] - datas[col].mean()) / datas[col].std()
        
    X_train, X_test, y_train, y_test = train_test_split(datas, target, test_size = 0.3)
    
    # transform to dict with list
    X_train_dict_list = X_train.to_dict(orient = 'list')
    X_test_dict_list = X_test.to_dict(orient = 'list')
    
    # make tf dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train_dict_list, y_train))
    train_ds = train_ds.shuffle(256).batch(128)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    valid_ds = tf.data.Dataset.from_tensor_slices((X_test_dict_list, y_test))
    valid_ds = valid_ds.batch(128)
    
    model.fit(ds, epochs = 3, validation_data = valid_ds)
    
'''