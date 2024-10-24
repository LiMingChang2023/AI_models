import tensorflow as tf
from keras import layers, models
import numpy as np

class attention(layers.Layer):
    
    def __init__(self, num_head, d_model, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head

        assert d_model % num_head == 0, 'd_model must divided by num_head'
        self.head_dim = d_model // num_head

        self.query = layers.Dense(d_model, use_bias=False)
        self.key = layers.Dense(d_model, use_bias=False)
        self.value = layers.Dense(d_model, use_bias=False)
        self.fc = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)


    def split_head(self, x, ):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_head, self.head_dim))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x

    def mask(self, seqlen):
        mask = tf.ones((seqlen, seqlen), dtype=tf.float32)
        mask = tf.linalg.band_part(mask, -1, 0)

        return mask

    def call(self, keys, queries, values, use_mask=False, training=False):
        batch_size = tf.shape(keys)[0]
        seqlen = tf.shape(keys)[1]
        
        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)

        queries = self.split_head(queries)
        keys = self.split_head(keys)
        values = self.split_head(values)
        
        qk = tf.einsum('nhqd, nhkd->nhqk', queries, keys)
        
        if use_mask:
            mask = self.mask(seqlen)
            mask = tf.expand_dims(mask, 0)
            mask = tf.expand_dims(mask, 1)
            qk = qk + (1 - mask) * -1e9

        scores = tf.nn.softmax(qk / tf.math.sqrt(tf.cast(self.head_dim, tf.float32)), axis=-1)
        if training:
            scores = self.dropout(scores, training=training)

        out = tf.einsum('nhqk, nhvd->nhqd', scores, values)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, seqlen, self.d_model))
        
        output = self.fc(out)

        return output

def position_encoding(seqlen, d_model, times=10000):

    pos = np.arange(seqlen)[:, np.newaxis]   
    depths = np.arange(d_model)[np.newaxis, :]
    depths = 2 * ((depths) // 2) / d_model

    angle_rates = 1 / times**depths
    angle_rads = pos * angle_rates

    pos_encoding = np.zeros((seqlen, d_model))
    pos_encoding[:, 0::2] = np.sin(angle_rads)[:, 0::2]
    pos_encoding[:, 1::2] = np.cos(angle_rads)[:, 1::2]

    return tf.cast(pos_encoding, tf.float32)


class PositionEmbedding(layers.Layer):
    
    def __init__(self, seqlen, d_model, vocal_size, embed=False):
        super().__init__()
        
        self.d_model = d_model
        self.pos_encoding = position_encoding(seqlen, d_model)
        self.embedding = layers.Embedding(vocal_size, d_model, mask_zero=True) if embed else lambda x: x

    def call(self, x):
        seqlen = x.shape[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model), tf.float32)

        x = x + self.pos_encoding[tf.newaxis, :seqlen, :]

        return x

class attentionLayer(layers.Layer):

    def __init__(self, num_head, d_model, dropout=0.0):
        super().__init__()
        self.attention = attention(num_head, d_model, dropout=dropout)
        self.add_1 = layers.Add()
        self.ln_1 = layers.LayerNormalization(epsilon=1e-08)

        self.ffn_1 = layers.Dense(d_model * 4, activation='relu')  # Typically expanded size
        self.ffn_2 = layers.Dense(d_model)  # Output layer to match d_model
        
        self.add_2 = layers.Add()
        self.ln_2 = layers.LayerNormalization(epsilon=1e-08)

    def call(self, query, key, value, use_mask=False, training=False):
        
        attn_out = self.attention(query, key, value, use_mask=use_mask, training=training)
        x = self.add_1([query, attn_out])
        x = self.ln_1(x)
        
        ffn_out = self.ffn_1(x)
        ffn_out = self.ffn_2(ffn_out)
        
        x = self.add_2([ffn_out, x])
        out = self.ln_2(x)

        return out 
    

class Resblock(layers.Layer):
    
    def __init__(self, filters: int, kernel_size: tuple, strides: int):
        super().__init__()
        
        self.conv1 = layers.Conv2D(filters, kernel_size, 1, padding='same')
        self.bn1 = layers.BatchNormalization(epsilon=1e-08)
        self.relu1 = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filters, kernel_size, strides, padding='same')
        self.bn2 = layers.BatchNormalization(epsilon=1e-08)
        
        self.filters = filters
        self.shortcut = layers.Conv2D(filters, 1, strides, padding='same')

        self.add = layers.Add()
        self.end_relu = layers.Activation('relu')
        
    def call(self, x):
        
        shortcut = self.shortcut(x) if x.shape[-1] != self.filters else x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        out = self.add([x, shortcut])
        out = self.end_relu(out)

        return out

        

        
        

