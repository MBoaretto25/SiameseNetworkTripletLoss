import warnings

import numpy as np
from tensorflow.keras.callbacks import Callback


class SaveEmbeddingModel(Callback):
    def __init__(self, filepath, emb_model, save_weights_only=True, monitor='loss', verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self._emb_model = emb_model
        self.best = np.Inf
        self._save_weights_only = save_weights_only
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("SaveEmbeddingModelWeights requires %s available!" % self.monitor, RuntimeWarning)

        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        if self.verbose == 1:
            print("Saving embedding model weights at %s" % filepath)

        if self._save_weights_only:
            self._emb_model.save_weights(filepath, overwrite=True)
        else:
            self._emb_model.save(filepath, overwrite=True)

