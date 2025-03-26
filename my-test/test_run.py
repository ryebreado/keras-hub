import keras
import keras_hub
from keras_hub.src.models.causal_lm import CausalLM
import numpy as np
import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras_hub.src.tokenizers.unicode_codepoint_tokenizer import UnicodeCodepointTokenizer
from keras_hub.src.models.language_classifier.language_classifier_backbone import (
    LanguageClassifier,
    LanguageClassifierBackbone
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
print(y_train)
# Initialize tokenizer and train it on your data
vocab_size = 1000  # Adjust based on your dataset size
# Adjust the tokenizer initialization

# Check if you need to specify required parameters
tokenizer = UnicodeCodepointTokenizer()

# Tokenize data
X_train_tokens = tokenizer(X_train)
# X_train_tokens = np.array(X_train_tokens, dtype=np.float32)
X_test_tokens = tokenizer(X_test)
# X_test_tokens = np.array(X_test_tokens, dtype=np.float32)


# Find the length of the longest sequence
max_length = max(len(seq) for seq in X_train_tokens)

# Pad all sequences to the same length
X_train_padded = pad_sequences(X_train_tokens,
                               maxlen=max_length,
                               padding='post',  # Add padding at the end
                               value=0)         # Pad with zeros

# Convert to numpy array
X_train_tokens = np.array(X_train_padded, dtype=np.float32)

# Create backbone and model
backbone = LanguageClassifierBackbone(
    vocab_size=tokenizer.vocabulary_size(),
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
# Assuming your labels are text like "English", "French", etc.
# First convert to numerical indices
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train)

# For categorical crossentropy loss, one-hot encode:
y_train = tf.keras.utils.to_categorical(y_encoded)

# Train the model
model.fit(
    X_train_tokens,
    np.array(y_train),
    epochs=10,
    batch_size=16,
    validation_data=(X_test_tokens, np.array(y_test))
)
