import numpy as np
import pickle as pkl
import os
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import keras
from tensorflow.python.ops import math_ops
import keras.backend as K

input_shape = (60, 40)
out_steps = 60
out_cat = 51


class KerasDataGenerator(keras.utils.all_utils.Sequence):

    def __init__(self, pickle_path, batch_size_x_16=1, shuffle=True):
        self.batch_size = batch_size_x_16
        self.shuffle = shuffle

        # This is where the data lives
        self.path = pickle_path
        self.files = None

        # init
        self.on_epoch_end()

    def __len__(self):
        """
        The number of batches per epoch is the total number of pickles / bath_size_x_16
        """
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        """
        This is the actual generator; one batch of data
        """
        # Generate indexes of the batch
        fs = self.files[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(fs)
        return X, y

    def on_epoch_end(self):
        """
        Epoch init consists of generating a list of the pickle files and shuffling them if shuffle is True
        """
        self.files = [f for f in os.listdir(self.path) if f.endswith(".pkl")]

        if self.shuffle:
            np.random.shuffle(self.files)

    def __data_generation(self, files):
        """
        Generate a single batch based on the file names in the list files
        """
        # Initialization
        data = [pkl.load(open(os.path.join(self.path, f), "rb")) for f in files]
        X = np.concatenate([d[0] for d in data], axis=0)
        y = np.concatenate([d[1] for d in data], axis=0)
        X = np.concatenate((X[:, :, :5], X[:, :, 10:15], X[:, :, 21:]), axis=2)
        y = y[:, :, :]

        return X, list(y.transpose(1, 0, 2))

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_ts_model(input_shape, output_steps, n_classes,
                   head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    outputs = [layers.Dense(n_classes, activation="softmax")(x) for _ in range(output_steps)]
    return keras.Model(inputs, outputs)


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        SAE_vals = []

        for x, y in validation_generator:
            y_hat = self.model(x)
            y_hat = np.vstack(y_hat).reshape(64, 60, 51)

            y = np.vstack(y).reshape(64, 60, 51)

            SAE = np.abs(y - y_hat).sum(axis=2).mean()
            SAE_vals.append(SAE)

        print('Hezi hamelech')
        SAE_total = np.array(SAE_vals).mean()
        print('The SAE value is:', SAE)







def custom_loss(y_true, y_pred):
    diff = K.abs(y_pred - y_true)
    diff1 = K.abs(y_pred[:,:4] - y_true[:,:4])
    loss = K.mean(diff, axis=-1) #mean over last dimension
    loss1 = K.mean(diff1, axis=-1)
    return loss + loss1

#warnings.simplefilter('ignore')
model = build_ts_model(input_shape, output_steps=out_steps, n_classes=out_cat,
                       head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate= 1e-4,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

model.compile(
    loss=custom_loss,
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["mse"],
    run_eagerly= True
)



# from loader import KerasDataGenerator
pickle_path_training = '/scratch/eliransc/rnn_data/training_trans_moments_fixed' #'/scratch/eliransc/new_gt_g_1_trans4/' #  Ichilov_gt_g_1_folders
pickle_path_valid = '/scratch/eliransc/rnn_data/moment_fix_test_set_1_a'

training_generator = KerasDataGenerator(pickle_path_training, batch_size_x_16=4)
validation_generator = KerasDataGenerator(pickle_path_valid, batch_size_x_16=4)
callbacks1 = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True), CustomCallback()  ] #

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    callbacks=callbacks1,
                    use_multiprocessing=True,
                    workers=10,
                    epochs=25,
                    verbose=0)


model.save('stupid_model.keras')

#warnings.simplefilter('ignore')
model1 = build_ts_model(input_shape, output_steps=out_steps, n_classes=out_cat,
                       head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25)

model1 = tf.keras.models.load_model('stupid_model.keras', custom_objects={'custom_loss': custom_loss })

res = model1.predict(validation_generator)

pkl.dump(res, open('res_trial.pkl', 'wb'))
