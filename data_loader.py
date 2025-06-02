import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

class LungCancerDataLoader:
    def __init__(self, data_dir, img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.stage_mapping = {
            'normal': -1,  # No cancer
            'stage0': 0,
            'stage1': 1,
            'stage2': 2,
            'stage3': 3,
            'stage4': 4
        }

    def preprocess_image(self, image_path):
        """Load and preprocess a single image."""
        img = load_img(image_path, target_size=self.img_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0,1]
        return img_array

    def create_dataset(self, split='train', batch_size=32):
        """Create dataset for training/validation/testing."""
        split_dir = os.path.join(self.data_dir, split)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist")
            
        images = []
        cancer_labels = []  # Binary: cancer or no cancer
        stage_labels = []   # Multi-class: stages 0-4
        
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            stage = self.stage_mapping.get(class_name)
            if stage is None:
                continue
                
            print(f"Processing {class_name} in {split} split...")
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                    continue
                    
                img_path = os.path.join(class_dir, img_name)
                try:
                    img_array = self.preprocess_image(img_path)
                    images.append(img_array)
                    
                    # Create labels for both tasks
                    has_cancer = 1 if stage >= 0 else 0
                    cancer_labels.append(has_cancer)
                    
                    # For stage classification, use -1 for no cancer, 0-4 for stages
                    stage_labels.append(max(0, stage))  # Convert -1 to 0 for no cancer
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue

        if not images:
            raise ValueError(f"No images found in {split_dir}")

        print(f"Found {len(images)} images in {split} split")
        
        # Convert to numpy arrays
        X = np.array(images)
        y_cancer = to_categorical(cancer_labels, num_classes=2)
        y_stage = to_categorical(stage_labels, num_classes=5)

        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            X,
            {
                'cancer_detection': y_cancer,
                'cancer_staging': y_stage
            }
        ))

        # Add data augmentation for training
        if split == 'train':
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and prefetch
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset

    def _augment(self, image, labels):
        """Apply data augmentation to training images."""
        # Random flips
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        
        # Random brightness and contrast
        image = tf.image.random_brightness(image, 0.3)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Random rotation
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        
        # Random zoom
        shape = tf.shape(image)
        zoom = tf.random.uniform(shape=[], minval=0.8, maxval=1.2)
        new_height = tf.cast(tf.cast(shape[0], tf.float32) * zoom, tf.int32)
        new_width = tf.cast(tf.cast(shape[1], tf.float32) * zoom, tf.int32)
        image = tf.image.resize(image, [new_height, new_width])
        image = tf.image.resize_with_crop_or_pad(image, shape[0], shape[1])
        
        # Random noise
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.1)
        image = image + noise
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, labels

# Example usage
if __name__ == '__main__':
    # Initialize data loader
    data_loader = LungCancerDataLoader('data')
    
    # Create datasets
    train_dataset = data_loader.create_dataset('train', batch_size=32)
    val_dataset = data_loader.create_dataset('val', batch_size=32)
    test_dataset = data_loader.create_dataset('test', batch_size=32) 