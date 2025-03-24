from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.language_classifier.language_classifier_backbone import LanguageClassifierBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from typing import Union

from keras_hub.src.utils.preset_utils import load_json

class LanguageClassifierLM(CausalLM):
    """Language classifier model that follows the CausalLM interface."""

    def __init__(
            self,
            backbone: LanguageClassifierBackbone,
            tokenizer: BytePairTokenizer,
            language_labels: list[str],
            **kwargs,
    ):
        super().__init__(backbone=backbone, tokenizer=tokenizer, **kwargs)
        self.language_labels = language_labels

    def call(self, inputs):
        logits = self.backbone(inputs)
        return {"logits": logits}

    def classify(self, text: Union[str, list[str]]):
        """Classify the language of input text."""
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

    # Override generate method since this model doesn't generate text
    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            "This model is a language classifier and does not support text generation."
        )

    @classmethod
    def from_pretrained(cls, directory: str):
        """Load model from directory."""
        # Load config
        config = load_json(f"{directory}/config.json")

        # Load language labels
        language_labels = load_json(f"{directory}/language_labels.json")

        # Load tokenizer
        tokenizer = BytePairTokenizer.from_pretrained(directory)

        # Create backbone and load weights
        backbone = LanguageClassifierBackbone(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_languages=config["num_languages"],
        )
        backbone.load_weights(f"{directory}/backbone_weights.h5")

        # Create model
        model = cls(backbone=backbone, tokenizer=tokenizer, language_labels=language_labels)
        return model