import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

class PreprocessingLayer(layers.Layer):
    def call(self, inputs):
        return (inputs - 0.5) * 2  # Scale to [-1, 1]

def create_model(input_shape=(224, 224, 3)):
    # Load pre-trained MobileNetV2 model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the model
    inputs = layers.Input(shape=input_shape)
    
    # Preprocess input using custom layer
    x = PreprocessingLayer()(inputs)
    
    # Pass through base model
    x = base_model(x)
    
    # Add custom layers with stronger regularization
    x = layers.GlobalAveragePooling2D()(x)
    
    # Shared features
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.7)(x)
    
    # Cancer detection branch
    cancer_detection = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)
    cancer_detection = layers.BatchNormalization()(cancer_detection)
    cancer_detection = layers.Dropout(0.7)(cancer_detection)
    cancer_detection = layers.Dense(2, activation='softmax', name='cancer_detection')(cancer_detection)
    
    # Cancer staging branch
    cancer_staging = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)
    cancer_staging = layers.BatchNormalization()(cancer_staging)
    cancer_staging = layers.Dropout(0.7)(cancer_staging)
    cancer_staging = layers.Dense(5, activation='softmax', name='cancer_staging')(cancer_staging)
    
    # Create model
    model = Model(inputs=inputs, outputs=[cancer_detection, cancer_staging])
    
    # Compile model with modern optimizer and adjusted loss weights
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Even lower initial learning rate
    
    model.compile(
        optimizer=optimizer,
        loss={
            'cancer_detection': 'categorical_crossentropy',
            'cancer_staging': 'categorical_crossentropy'
        },
        loss_weights={
            'cancer_detection': 0.4,  # Reduce weight of cancer detection
            'cancer_staging': 0.6     # Increase weight of staging
        },
        metrics={
            'cancer_detection': ['accuracy'],
            'cancer_staging': ['accuracy']
        }
    )
    
    return model

# Create the model instance
model = create_model() 