import keras
import keras_hub
from keras_hub.src.models.causal_lm import CausalLM
import numpy as np
import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_hub.src.models.language_classifier.language_classifier_backbone import (
    LanguageClassifier,
    LanguageClassifierBackbone
)
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer

# Example languages and sample sentences
language_samples = {
    "english": ["Hello, how are you?", "The weather is nice today.", "I love programming in Python."],
    "spanish": ["Hola, ¿cómo estás?", "El clima es agradable hoy.", "Me encanta programar en Python."],
    "french": ["Bonjour, comment ça va?", "Le temps est agréable aujourd'hui.", "J'aime programmer en Python."],
    "german": ["Hallo, wie geht es dir?", "Das Wetter ist heute schön.", "Ich liebe es, in Python zu programmieren."],
    "italian": ["Ciao, come stai?", "Il tempo è bello oggi.", "Adoro programmare in Python."],
}

# Prepare dataset
texts = []
labels = []
language_labels = list(language_samples.keys())

for i, lang in enumerate(language_labels):
    for text in language_samples[lang]:
        texts.append(text)
        labels.append(i)

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Initialize tokenizer and train it on your data
vocab_size = 1000  # Adjust based on your dataset size
# Adjust the tokenizer initialization

# Check if you need to specify required parameters
tokenizer = BytePairTokenizer(
    # vocab_size=vocab_size,
    # merges_file=None,  # This will be created during training
    # vocab_file=None,   # This will be created during training
    # add_special_tokens=True  # Include default special tokens
)
tokenizer.train_from_texts(X_train)

# Tokenize data
X_train_tokens = tokenizer.encode(X_train)
X_test_tokens = tokenizer.encode(X_test)

# Create backbone and model
backbone = LanguageClassifierBackbone(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=128,
    num_languages=len(language_labels),
)

model = LanguageClassifier(
    backbone=backbone,
    tokenizer=tokenizer,
    language_labels=language_labels,
)

# Compile the model
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"],
)

# Train the model
model.fit(
    X_train_tokens,
    np.array(y_train),
    epochs=10,
    batch_size=16,
    validation_data=(X_test_tokens, np.array(y_test))
)
