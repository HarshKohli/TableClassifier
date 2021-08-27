# Author: Harsh Kohli
# Date Created: 19-08-2021

import yaml
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from models import PointingNetworkDecoder
from utils.training_utils import train_epoch_ptrnet
from utils.data_utils import load_preprocessed_data

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
config = yaml.safe_load(open('config.yml', 'r'))
tf.config.run_functions_eagerly(config['execute_greedily'])

batch_size = config['ptrnet_batch_size']
model_name = config['ptrnet_model_name']
data = load_preprocessed_data(config['preprocessed_ptrnet_data'])

print('here')

model = PointingNetworkDecoder(config)
optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
save_path = os.path.join(config['model_dir'], model_name)
log_path = os.path.join(config['log_dir'], model_name)
dim = config['dim']


@tf.function
def train_step(query, tables, labels):
    with tf.GradientTape() as tape:
        loss = model([query, tables, labels], training=True)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        (grad, var)
        for (grad, var) in zip(gradients, model.trainable_variables)
        if grad is not None
    )
    return loss


train_batches, train_index_to_vec = data['train_batches'], data['train_index_to_vec']
dev_batches, dev_index_to_vec = data['dev_batches'], data['dev_index_to_vec']
test_batches, test_index_to_vec = data['test_batches'], data['test_index_to_vec']

train_iterations = len(train_batches)
print('Starting Training...')
for epoch_num in range(config['num_epochs']):
    print('Starting Epoch: ' + str(epoch_num))
    train_epoch_ptrnet(train_batches, train_index_to_vec, train_iterations, train_step)
