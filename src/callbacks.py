from deepsense import neptune

from keras.callbacks import Callback

ctx = neptune.Context()


def neptune_monitor_lgbm(channel_prefix=''):
    def callback(env):
        for name, loss_name, loss_value, _ in env.evaluation_result_list:
            if channel_prefix != '':
                channel_name = '{}_{}_{}'.format(channel_prefix, name, loss_name)
            else:
                channel_name = '{}_{}'.format(name, loss_name)
            ctx.channel_send(channel_name, x=env.iteration, y=loss_value)

    return callback


class NeptuneMonitor(Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.ctx = neptune.Context()
        # self.batch_loss_channel_name = '{} Batch Log-loss training'.format(self.model_name)
        self.epoch_loss_channel_name = '{} Log-loss training'.format(self.model_name)
        self.epoch_val_loss_channel_name = '{} Log-loss validation'.format(self.model_name)

        self.epoch_id = 0
        self.batch_id = 0

    def on_batch_end(self, batch, logs={}):
        self.batch_id += 1
        # self.ctx.channel_send(self.batch_loss_channel_name, self.batch_id, logs['loss'])

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1
        self.ctx.channel_send(self.epoch_loss_channel_name, self.epoch_id, logs['loss'])
        self.ctx.channel_send(self.epoch_val_loss_channel_name, self.epoch_id, logs['val_loss'])
