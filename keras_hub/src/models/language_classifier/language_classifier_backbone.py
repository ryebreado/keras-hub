import tensorflow as tf
import keras
from keras import layers
# from typing import Any, Dict, Optional, Tuple, Union
from typing import Union

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


class LanguageClassifierBackbone(Backbone):
    """Linear language classification backbone."""

    def __init__(
            self,
            vocab_size: int = 30000,
            embedding_dim: int = 256,
            num_languages: int = 10,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_languages = num_languages

        # Simple embedding layer followed by pooling and a linear layer
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        )
        self.global_pooling = layers.GlobalAveragePooling1D()
        self.classifier = layers.Dense(num_languages)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.global_pooling(x)
        return self.classifier(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_languages": self.num_languages,
        })
        return config


class LanguageClassifier(keras.Model):
    """Model for classifying the language of text inputs."""

    def __init__(
            self,
            backbone: LanguageClassifierBackbone,
            tokenizer: BytePairTokenizer,
            language_labels: list[str],
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.language_labels = language_labels

    def call(self, inputs):
        logits = self.backbone(inputs)
        return {"logits": logits}

    def predict_language(self, text: Union[str, list[str]]):
        """Predict the language of input text."""
        # Handle single string input
        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        # Tokenize input
        tokens = self.tokenizer.encode(text)

        # Get logits from model
        logits = self.backbone(tokens)

        # Get predictions
        predictions = tf.argmax(logits, axis=-1)

        # Map to language labels
        languages = [self.language_labels[pred] for pred in predictions.numpy()]

        # Return single result if input was a single string
        if is_single:
            return languages[0]
        return languages
