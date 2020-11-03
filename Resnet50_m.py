import tensorflow as tf

def build_Resnet50():
    model = tf.keras.applications.ResNet50(
    include_top=True, weights='imagenet', input_tensor=None, input_shape=[None, None, 3],
    pooling=None,)
    Resnet50_m = tf.keras.models.model_from_json(model.to_json())
    Resnet50_m.set_weights(model.get_weights())

    return Resnet50_m


class Res50(tf.keras.models.Model):
    def __init__(self, **kwargs):
        inception = build_Resnet50()
        super(Res50, self).__init__(inputs=inception.inputs, outputs=inception.outputs, **kwargs)


if __name__ == '__main__':
    build_Resnet50()
