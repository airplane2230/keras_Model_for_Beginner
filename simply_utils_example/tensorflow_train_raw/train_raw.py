from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf

from tqdm import tqdm

from lr_schedule import LRSchedule

print(tf.__version__)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def make_datasets(x, y):
    # (28, 28) -> (28, 28, 1)
    def _new_axis(x, y):
        y = tf.one_hot(y, depth = 10)
        
        return x[..., tf.newaxis], y
            
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(_new_axis, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(100).batch(32) # 배치 크기 조절하세요
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    return ds
    
train_ds = make_datasets(x_train, y_train)

# simple CNN model
def get_model():
    def get_preprocess():
        # rescaling, 1 / 255
        preprocessing_layer = tf.keras.models.Sequential([
                tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
            ])
        
        return preprocessing_layer
    
    preprocessing_layer = get_preprocess()
    
    inputs = Input(shape = (28, 28, 1))
    preprocessing_inputs = preprocessing_layer(inputs)
    
    x = Conv2D(filters = 32, kernel_size = (3, 3), activation='relu')(preprocessing_inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(filters = 64, kernel_size =(3, 3), activation='relu')(x)
    
    x = Flatten()(x)
    x = Dense(64, activation = 'relu')(x)
    outputs = Dense(10, activation = 'softmax')(x)
    
    model = Model(inputs = inputs, outputs = outputs)
    
    return model

model = get_model()
# model.summary()

# hyperparameter
init_epoch = 1
EPOCHS = 1000
warmup_epoch = int(EPOCHS * 0.3)
init_lr = 0.1
min_lr = 1e-6
power = 1.

lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate = init_lr,
    decay_steps = EPOCHS - warmup_epoch,
    end_learning_rate = min_lr,
    power = power
)

optimizer = tf.keras.optimizers.Adam(learning_rate = LRSchedule(init_lr,
                                                                warmup_epoch=warmup_epoch,
                                                                decay_fn=lr_scheduler
                                                                ))

init_path = './model/model_ckpt/init'
init_ckpt = tf.train.Checkpoint(model = model, optimizer = optimizer)

init_ckpt_manager = tf.train.CheckpointManager(init_ckpt, init_path, max_to_keep = 10)
init_ckpt_manager.save()
print('save init!')

@tf.function
def train_step(inp, tar, training = True):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = loss_function(tar, predictions)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy.update_state(tar, predictions)
    
    return loss, predictions

# Train! 
loss_plot = []
loss_function = tf.keras.losses.CategoricalCrossentropy()

train_accuracy = tf.keras.metrics.CategoricalAccuracy()

ckpt_path = './model/model_ckpt/'
ckpt = tf.train.Checkpoint(epoch = tf.Variable(1), loss = tf.Variable(1., dtype = tf.float64),
                           accuracy = tf.Variable(1., dtype = tf.float64),
                           model = model,
                           optimizer = optimizer
                           )
ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep = 10)

ckpt.restore(ckpt_manager.latest_checkpoint)
if ckpt_manager.latest_checkpoint:
    print(f'Restored from {ckpt_manager.latest_checkpoint}')
    print(f'epoch: {ckpt.epoch}, accuracy: {ckpt.accuracy}, loss: {ckpt.loss}')
else:
    print('Initializing from Scratch')
    
for epoch in range(init_epoch, EPOCHS):
    total_loss = 0.
    train_accuracy.reset_states()
    
    tqdm_dataset = tqdm(enumerate(train_ds))
    
    LR_for_print = LRSchedule(init_lr,warmup_epoch=warmup_epoch,decay_fn=lr_scheduler)
    
    print('current learning_rate: ', LR_for_print(epoch).numpy())
    
    for (batch, (tensor, target)) in tqdm_dataset:
        batch_loss, predictions = train_step(tensor, target, training = True)
        total_loss += batch_loss
        
        total_loss_format = total_loss / (batch + 1)
        
        tqdm_dataset.set_postfix({
            'Epoch': epoch,
            'Loss': '{:06f}'.format(batch_loss.numpy()),
            'Total Loss': '{:06f}'.format(total_loss_format),
            'Accuracy': '{:04f}'.format(train_accuracy.result().numpy())
        })
        
    loss_plot.append(total_loss_format)
    
    if np.min(loss_plot) == loss_plot[-1]:
        print(f'{epoch} - min Loss!: {loss_plot[-1]}, and Accuracy: {train_accuracy.result().numpy()}')
        ckpt.epoch.assign(epoch)
        ckpt.accuracy.assign(train_accuracy.result().numpy())
        ckpt.loss.assign(total_loss_format)
        ckpt_manager.save()