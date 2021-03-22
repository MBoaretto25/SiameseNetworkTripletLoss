import os
# Disable Tensorflow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from datetime import datetime

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model

from batch_generator import BatchGenerator
from callbacks import SaveEmbeddingModel
from losses import triplet_loss
from utils import load_samples, enable_memory_growth, load_yaml


print("Tensorflow Version : {}".format(tf.__version__))


enable_memory_growth()

cfg = load_yaml("configs/facenet_triplet_mixed.yaml")

# Load images
dataset_folder = "datasets"
images_path = os.path.join(dataset_folder, "ms1m_align_112/imgs_mixed/")

X = load_samples(images_path, 0)
counter = 0
for k, v in X.items():
    counter += len(v)
print("============================")
print('{} Images loaded within {} classes'.format(counter, len(X.keys())) )
print("============================")

generator_config = {
    "IMAGE_W": cfg['input_size'],
    "IMAGE_H": cfg['input_size'],
    "IMAGE_C": 3,
    "BATCH_SIZE": cfg['batch_size'],
    "DATASET_PATH": images_path,
    "EMB_SIZE": cfg['embd_shape'],
    "CUT_FACE": cfg['is_ccrop'],
    "DEBUG": False
}

gen = BatchGenerator(X, generator_config, shuffle=False, jitter=False)

## Build Triplet Model
input_shape = (cfg['input_size'], cfg['input_size'], 3)

backend_model = load_model("facenet/facenet_keras.h5")
backend_model.load_weights("facenet/weights/facenet_keras_weights.h5")

A = Input(shape=backend_model.input_shape[1:], name='anchor')
P = Input(shape=backend_model.input_shape[1:], name='anchorPositive')
N = Input(shape=backend_model.input_shape[1:], name='anchorNegative')

enc_A = backend_model(A)
enc_P = backend_model(P)
enc_N = backend_model(N)

# Model
tripletModel = Model([A, P, N], [enc_A, enc_P, enc_N])
tripletModel.compile(optimizer='adam', loss=triplet_loss(cfg['alpha']))


## CallBacks
if not os.path.isdir("checkpoints"):
    os.mkdir("checkpoints")

best_loss_model_name = "{}_{}".format(cfg['sub_name'], datetime.now().strftime(format="%Y%m%d%H%M"))
best_loss_path = os.path.join("checkpoints", best_loss_model_name)

if not os.path.isdir(best_loss_path):
    os.mkdir(best_loss_path)


check_point_saver_best_loss = ModelCheckpoint(os.path.join(best_loss_path, "triplet_face_loss.h5"), monitor='loss',
                                              verbose=1,
                                              save_best_only=True, save_weights_only=False, mode='auto',
                                              period=1)

check_point_saver = ModelCheckpoint(os.path.join(best_loss_path, "triplet_face_ckp.h5"), monitor='loss',
                                    verbose=1,
                                    save_best_only=False, save_weights_only=False, mode='auto',
                                    period=1)

tb = TensorBoard(log_dir=os.path.join(best_loss_path, "logs"), histogram_freq=0, batch_size=cfg['batch_size'],
                 write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
                 embeddings_layer_names=None, embeddings_metadata=None)


# Save the embedding model weights if you save a new snn best model based on the model checkpoint above
emb_weight_saver = SaveEmbeddingModel(os.path.join(best_loss_path, 'emb_model_weights.h5'),
                                      emb_model=backend_model,
                                      save_weights_only=False)

dataset_len = cfg['num_samples']
steps_per_epoch = dataset_len // cfg['batch_size']

tripletModel.fit_generator(gen,
                           epochs=cfg['epochs'],
                           steps_per_epoch=steps_per_epoch,
                           verbose=1,
                           workers=10,
                           max_queue_size=40,
                           callbacks=[check_point_saver_best_loss, tb, check_point_saver, emb_weight_saver]
                           )

print("[*] training done!")
