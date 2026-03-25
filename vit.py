import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_h, self.patch_w = patch_size
        self.proj = tf.keras.layers.Dense(embed_dim)

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_h, self.patch_w, 1],
            strides=[1, self.patch_h, self.patch_w, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(patches, [tf.shape(images)[0], -1, patches.shape[-1]])
        return self.proj(patches)

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim
        })
        return config

@register_keras_serializable()
class AddPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_dim, max_len=500, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.pos_emb = self.add_weight(
            shape=(1, max_len, embed_dim),
            initializer="random_normal"
        )

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_emb[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "max_len": self.max_len
        })
        return config

@register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )
        self.norm2 = tf.keras.layers.LayerNormalization()

        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation='gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embed_dim)
        ])

    def call(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout
        })
        return config

@register_keras_serializable()
class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        num_classes,
        patch_size,
        embed_dim=128,
        depth=6,
        num_heads=2,
        mlp_dim=256,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        self.pos_embed = AddPositionEmbedding(embed_dim)
        self.embed_dropout = tf.keras.layers.Dropout(0.1)

        self.transformer = [
            TransformerBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(depth)
        ]

        self.norm = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.embed_dropout(x, training=False)

        for block in self.transformer:
            x = block(x)

        x = self.norm(x)
        x = tf.reduce_mean(x, axis=1)
        return self.head(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        


model = VisionTransformer(
    num_classes=12,  
    patch_size=(4, 8),
    embed_dim=192,
    depth=8,
    num_heads=3,
    mlp_dim=384
)

"""
  tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    ),"""
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',  # dónde guardar
        monitor='val_accuracy',       # qué métrica vigilar
        save_best_only=True,          # solo guarda si mejora
        verbose=1
    )
]


