import keras
import keras_hub
from keras_hub.src.models.causal_lm import CausalLM
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from keras_hub.src.tokenizers.unicode_codepoint_tokenizer import UnicodeCodepointTokenizer
from keras_hub.src.models.language_classifier.language_classifier_backbone import (
    LanguageClassifier,
    LanguageClassifierBackbone
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import os
import psutil
import sys
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def load_tatoeba_data(sample_size=100, random_state=42):
    """Load and sample data from the Tatoeba sentences dataset.
    
    Args:
        sample_size: Number of rows to sample from the dataset
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with columns 'language' and 'text'
    """
    # Check if file exists
    file_path = "my-test/data/tatoeba/sentences.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Tatoeba dataset not found at {file_path}")
    
    # Load the dataset
    df = pd.read_csv(
        file_path,
        sep='\t',
        header=None,
        names=['text_id', 'language_id', 'text'],
        usecols=['language_id', 'text']  # We only need these columns
    )
    
    # Rename columns to match our expected format
    df = df.rename(columns={'language_id': 'language'})
    
    return df

def prepare_balanced_dataset(df, min_samples_per_lang=100, samples_per_lang=1000, random_state=42):
    """Prepare a balanced dataset with equal samples per language.
    
    Args:
        df: Input DataFrame
        min_samples_per_lang: Minimum number of samples required for a language to be included
        samples_per_lang: Number of samples to take from each language
        random_state: Random seed for reproducibility
        
    Returns:
        Balanced DataFrame with equal samples per language
    """
    # Count samples per language
    lang_counts = df['language'].value_counts()
    
    # Filter languages with sufficient samples
    valid_languages = lang_counts[lang_counts >= min_samples_per_lang].index
    print(f"Found {len(valid_languages)} languages with at least {min_samples_per_lang} samples")
    
    # Sample equal number of examples from each language
    balanced_samples = []
    for lang in valid_languages:
        lang_samples = df[df['language'] == lang].sample(
            n=min(samples_per_lang, len(df[df['language'] == lang])),
            random_state=random_state
        )
        balanced_samples.append(lang_samples)
    
    balanced_df = pd.concat(balanced_samples)
    
    # Print language distribution
    print("\nLanguage distribution in balanced dataset:")
    print(balanced_df['language'].value_counts())
    print(f"Total samples: {len(balanced_df)}")
    
    return balanced_df

def split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """Split dataset into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        train_df, val_df, test_df
    """
    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    # Split each language separately to maintain distribution
    for lang in df['language'].unique():
        lang_data = df[df['language'] == lang]
        
        # Calculate split sizes
        n = len(lang_data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        # Split the data
        train = lang_data.sample(n=train_size, random_state=random_state)
        remaining = lang_data.drop(train.index)
        val = remaining.sample(n=val_size, random_state=random_state)
        test = remaining.drop(val.index)
        
        train_dfs.append(train)
        val_dfs.append(val)
        test_dfs.append(test)
    
    # Combine splits
    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)
    
    # Print split statistics
    print("\nDataset Split Statistics:")
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print("\nLanguages in each split:")
    print("Training:", len(train_df['language'].unique()))
    print("Validation:", len(val_df['language'].unique()))
    print("Test:", len(test_df['language'].unique()))
    
    return train_df, val_df, test_df

# Load data from Tatoeba
try:
    print(f"Memory usage before loading data: {get_memory_usage():.2f} MB")
    df = load_tatoeba_data()
    print(f"Memory usage after loading data: {get_memory_usage():.2f} MB")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the Tatoeba dataset is available at my-test/data/tatoeba/sentences.csv")
    exit(1)

# Prepare balanced dataset
df = prepare_balanced_dataset(
    df,
    min_samples_per_lang=50000,
    samples_per_lang=100,
    random_state=42
)

# Split into train, validation, and test sets
train_df, val_df, test_df = split_dataset(
    df,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
)

# Print data distribution
print("\nFinal Data Distribution:")
print("Training set:")
print(train_df['language'].value_counts())
print("\nValidation set:")
print(val_df['language'].value_counts())
print("\nTest set:")
print(test_df['language'].value_counts())
print("-" * 50)

print(f"Memory usage before tokenization: {get_memory_usage():.2f} MB")

# Initialize tokenizer with vocabulary size clamping
tokenizer = UnicodeCodepointTokenizer(vocabulary_size=65536)

# Tokenize all sets
X_train_tokens = tokenizer(train_df['text'].tolist())
X_val_tokens = tokenizer(val_df['text'].tolist())
X_test_tokens = tokenizer(test_df['text'].tolist())

print(f"Memory usage after tokenization: {get_memory_usage():.2f} MB")

# Pad sequences to the same length
max_length = 100
X_train_padded = pad_sequences(X_train_tokens, maxlen=max_length, padding='post', value=0)
X_val_padded = pad_sequences(X_val_tokens, maxlen=max_length, padding='post', value=0)
X_test_padded = pad_sequences(X_test_tokens, maxlen=max_length, padding='post', value=0)

print(f"Memory usage after padding: {get_memory_usage():.2f} MB")

# Convert to numpy arrays of int32 type (for embedding layer)
X_train_tokens = np.array(X_train_padded, dtype=np.int32)
X_val_tokens = np.array(X_val_padded, dtype=np.int32)
X_test_tokens = np.array(X_test_padded, dtype=np.int32)

print("X_train_tokens", X_train_tokens[0, :10])
print("X_test_tokens", X_test_tokens[0, :10])

print(f"Memory usage after numpy conversion: {get_memory_usage():.2f} MB")

# Verify that all tokens are within vocabulary size
print("\nToken range check:")
print(f"Training set min: {X_train_tokens.min()}, max: {X_train_tokens.max()}")
print(f"Validation set min: {X_val_tokens.min()}, max: {X_val_tokens.max()}")
print(f"Test set min: {X_test_tokens.min()}, max: {X_test_tokens.max()}")
print("-" * 50)

# Convert language labels to numerical values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(train_df['language'])
y_val_encoded = label_encoder.transform(val_df['language'])
y_test_encoded = label_encoder.transform(test_df['language'])

print("y_train_encoded", y_train_encoded)

# Get unique languages for model initialization
language_labels = df['language'].unique().tolist()

# Print batch size information
batch_size = 32  # Reduced from 128 for better generalization
print(f"\nTraining Configuration:")
print(f"Batch size: {batch_size}")
print(f"Number of batches per epoch: {len(X_train_tokens) // batch_size}")
print(f"Training samples: {len(X_train_tokens)}")
print(f"Validation samples: {len(X_val_tokens)}")
print("-" * 50)

# Define callbacks for better training
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=2,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=1,
        min_lr=1e-5,
        verbose=1
    )
]

# Create backbone and model
backbone = LanguageClassifierBackbone(
    vocab_size=65536,
    embedding_dim=64,
    num_languages=len(language_labels),
    sequence_length=max_length,
)

model = LanguageClassifier(
    backbone=backbone,
    tokenizer=tokenizer,
    language_labels=language_labels,
)

# Print model summary
print("\nModel Architecture:")
model.summary()
print("-" * 50)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=["accuracy"]
)

# Train the model with validation data and callbacks
print("\nStarting training...")
history = model.fit(
    X_train_tokens,
    y_train_encoded,
    epochs=30,
    batch_size=batch_size,
    validation_data=(X_val_tokens, y_val_encoded),
    callbacks=callbacks,
    verbose=1
)

# Print training summary
print("\nTraining Summary:")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
print("-" * 50)

# Evaluate on test set
print("\nTest Set Evaluation:")

def run_test_eval():
    total_correct = 0
    total_samples = 0 
    for sample in test_df.iterrows():
        sample = sample[1]

        test_tokens = tokenizer([sample['text']])
        test_padded = pad_sequences(test_tokens, maxlen=max_length, padding='post', value=0)
        test_tokens = np.array(test_padded, dtype=np.int32)
        true_label =sample['language']
        
        predictions = model.predict(test_tokens, verbose=0)
        predicted_indices = tf.argmax(predictions["logits"], axis=-1)
        predicted_label = label_encoder.inverse_transform(predicted_indices)[0]

        total_correct += 1 if (predicted_label == true_label) else 0 
        total_samples+= 1
    
    accuracy = total_correct / total_samples
    
    print(f"Test set accuracy: {accuracy:.4f}")
        

run_test_eval()

# Take 10 random examples from the test set
test_samples = test_df.sample(n=10, random_state=42)

# Get predictions
test_tokens = tokenizer(test_samples['text'].tolist())
test_padded = pad_sequences(test_tokens, maxlen=max_length, padding='post', value=0)
test_tokens = np.array(test_padded, dtype=np.int32)

predictions = model.predict(test_tokens)
predicted_indices = tf.argmax(predictions["logits"], axis=-1)
predicted_probs = predictions["logits"]
max_probs = tf.reduce_max(predictions["logits"], axis=-1)

# Print results with detailed prediction information
print("\nTest Results:")
print("-" * 50)
for i, (text, true_lang, pred_idx, max_prob) in enumerate(zip(
    test_samples['text'], 
    test_samples['language'], 
    predicted_indices.numpy(),
    max_probs.numpy()
)):
    print(f"Example {i+1}:")
    print(f"Text: {text}")
    print(f"True Language: {true_lang}")
    print(f"Predicted Index: {pred_idx}")
    print(f"Predicted Language: {label_encoder.inverse_transform([pred_idx])[0] if pred_idx < len(language_labels) else 'OUT_OF_RANGE'}")
    print(f"Confidence: {max_prob:.4f}")
    
    # Print top 3 predictions
    top3_indices = tf.argsort(predicted_probs[i], direction='DESCENDING')[:3]
    print("Top 3 predictions:")
    for idx in top3_indices:
        lang = label_encoder.inverse_transform([idx])[0] if idx < len(language_labels) else 'OUT_OF_RANGE'
        prob = predicted_probs[i][idx]
        print(f"  {lang}: {prob:.4f}")
    
    print("-" * 50)