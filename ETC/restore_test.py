import tensorflow.compat.v1 as tf
from etcmodel.models import input_utils,modeling
#etcmodel=tf.keras.models.load_model("etcmodel/pretrained/model.ckpt")
#etcmodel=tf.saved_model.load("etcmodel/pretrained/model.ckpt")
model_config=input_utils.get_model_config(
        model_dir="./etcmodel/pretrained",
        source_file="./etcmodel/pretrained/etc_config.json",
        write_from_source=True)
etcmodel=modeling.EtcModel(model_config)
etcmodel.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=[tf.keras.metrics.BinaryAccuracy()])
load_status=etcmodel.load_weights("./etcmodel/pretrained/model.ckpt")
#
