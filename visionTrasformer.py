from modelComponent import attentionLayer
from keras import layers
import tensorflow as tf

class viT(layers.Layer):
    
    def __init__(self, num_head, num_layers, patch, filters):
        
        super().__init__()
        self.conv = layers.Conv2D(filters, patch, patch, padding='same')
        self.bn = layers.BatchNormalization(epsilon=1e-08)
        self.relu = layers.Activation('relu')
        self.patch = patch

        self.attn = [attentionLayer(num_head, filters) for _ in range(num_layers)]

        self.ffn_1 = layers.Dense(filters*4, activation='relu')
        self.ffn_2 = layers.Dense(filters)

    def split_patch(self, x): # (batch, height, weight, filters)
        batch_size, height, weight, filters = tf.shape(x)
        images = tf.reshape(x, (batch_size, \
                                height // self.patch, self.patch, \
                                weight // self.patch, self.patch, \
                                filters))
        
        images = tf.transpose(images, perm=[0, 1, 3, 2, 4, 5])
        images = tf.reshape(images, (batch_size, -1, self.patch * self.patch * filters))
        
        return images
    
    def call(self, images, training=False):
        x = self.conv(images)
        x = self.bn(x)
        x = self.relu(x)
        
        x = self.split_patch(x)
        for attn in self.attn:
            x = attn(x, mask=False, training=training)

        x = self.ffn_1(x)
        out = self.ffn_2(x)

        return out

        
        
        