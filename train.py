import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_loader import LungCancerDataLoader
import os

# Create necessary directories
os.makedirs('checkpoints', exist_ok=True)

# Initialize data loader
data_loader = LungCancerDataLoader('data')

# Create datasets with smaller batch size for better generalization
train_dataset = data_loader.create_dataset('train', batch_size=4)  # Reduced batch size
val_dataset = data_loader.create_dataset('val', batch_size=4)

# Load the model
from first import model

# Define callbacks with adjusted parameters
callbacks = [
    ModelCheckpoint(
        'checkpoints/best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=50,  # Increased patience
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,  # More gradual learning rate reduction
        patience=15,  # Increased patience
        min_lr=1e-8,  # Lower minimum learning rate
        verbose=1
    )
]

# Train the model with more epochs
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=1000,  # Increased epochs
    callbacks=callbacks
)

# Save the final model in the new Keras format
model.save('lung_cancer_model_final.keras') 