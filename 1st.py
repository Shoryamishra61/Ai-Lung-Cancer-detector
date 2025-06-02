import tensorflow as tf
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

# Two output branches:
# 1. Cancer Detection (binary: cancer/no cancer)
cancer_detection = Dense(2, activation='softmax', name='cancer_detection')(x)

# 2. Cancer Staging (5 classes: Stage 0, I, II, III, IV)
# Only relevant when cancer is detected
cancer_staging = Dense(5, activation='softmax', name='cancer_staging')(x)

# Create model with multiple outputs
model = Model(inputs=base_model.input, 
             outputs=[cancer_detection, cancer_staging])

# Freeze base model layers initially
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with different losses for each task
model.compile(optimizer='adam',
             loss={'cancer_detection': 'categorical_crossentropy',
                   'cancer_staging': 'categorical_crossentropy'},
             loss_weights={'cancer_detection': 1.0,
                          'cancer_staging': 0.5},
             metrics={'cancer_detection': ['accuracy'],
                     'cancer_staging': ['accuracy']})

# Print model summary
model.summary()