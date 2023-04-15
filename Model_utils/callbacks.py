import tensorflow as tf


class StopAt(tf.keras.callbacks.Callback):
    def __init__(self, key='val_accuracy', limit=.95):
        super().__init__()
        self._key = key
        self._limit = limit

    def on_epoch_end(self, epoch, logs=None):
        test_accuracy = logs.get(self._key)
        if test_accuracy > self._limit:
            self.model.stop_training = True

