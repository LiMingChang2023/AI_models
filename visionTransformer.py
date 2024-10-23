from modelComponent import attentionLayer
from keras import layers, models
import tensorflow as tf

class PosEmbedding(layers.Layer):
    def __init__(self, len, dim):
        super().__init__()

        self.pos_weight = self.add_weight(
            shape=(1, len, dim), 
            dtype=tf.float32, 
            initializer='random_normal', 
            trainable=True
        )
        
        self.add = layers.Add()
        
    def call(self, x):
        x = self.add([x, self.pos_weight])

        return x
        
# patch without overlap
class VisionTransformer(layers.Layer):
    
    def __init__(self, shape, num_head, num_layers, patch, _class=10):
        super().__init__()
        self.patch = patch        
        self.d_model = patch * patch * shape[-1]
        self.new_h, self.new_w = shape[0] // patch, shape[1] // patch

        self.pos_enc = PosEmbedding(self.new_h * self.new_w, self.d_model)
        self.attn = [attentionLayer(num_head, self.d_model) for _ in range(num_layers)]        
        # Add a classification token
        self.class_token = self.add_weight("class_token", shape=(1, 1, self.d_model), initializer="zeros", trainable=True)
        self.flatten = layers.Flatten()

        self.fc = self.clsBlock(3, _class)

    def patchEmbedding(self, x, batch):

        x = tf.reshape(x, (batch, self.new_h, self.patch, self.new_w, self.patch, -1))
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, (batch, self.new_h * self.new_w, -1))
        
        x = self.pos_enc(x)

        return x
    
    def clsBlock(self, layers, cls):
        model = models.Sequential()
        for _ in range(layers):
            model.add(layers.Dense(self.d_model * 4, activation='relu'))
            model.add(layers.Dense(self.d_model))

        model.add(layers.Dense(cls), activation='softmax')
        return model

    @tf.function
    def call(self, images, training=False):

        shape = tf.shape(images)
        x = self.patchEmbedding(images, shape[0])

        # Prepend the classification token
        class_tokens = tf.broadcast_to(self.class_token, (shape[0], 1, tf.shape(x)[-1]))
        x = tf.concat([class_tokens, x], axis=1)

        for attn in self.attn:
            x = attn(x, x, x, use_mask=False, training=training)

        x = self.ffn_1(x)
        x = self.ffn_2(x)

        # Use the output of the classification token
        class_output = x[:, 0, :]  # Get the class token's output
        out = self.flatten(class_output)
        output = self.fc(out)

        return output
