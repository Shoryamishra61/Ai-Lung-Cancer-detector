import tensorflow as tf
import numpy as np
from PIL import Image
import os

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image."""
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def predict_image(model, image_path):
    """Make prediction on a single image."""
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return None
    
    try:
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get cancer detection result
        cancer_prob = predictions[0][0][1]  # Probability of cancer
        cancer_result = "Cancer Detected" if cancer_prob > 0.5 else "No Cancer Detected"
        
        # Get cancer stage
        stage_probs = predictions[1][0]
        stage = np.argmax(stage_probs)
        stage_prob = stage_probs[stage]
        
        stages = ["Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"]
        stage_result = stages[stage] if cancer_prob > 0.5 else "N/A"
        
        return {
            "cancer_detection": cancer_result,
            "cancer_probability": f"{cancer_prob:.2%}",
            "cancer_stage": stage_result,
            "stage_probability": f"{stage_prob:.2%}"
        }
    except Exception as e:
        print(f"Error making prediction for {image_path}: {e}")
        return None

def main():
    # Load the trained model
    try:
        print("Loading model...")
        model = tf.keras.models.load_model('lung_cancer_model_final.keras')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Directory containing test images
    test_dir = 'raw_images/stage1'
    
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found!")
        return
    
    # Process each image in the directory
    for image_name in os.listdir(test_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            image_path = os.path.join(test_dir, image_name)
            print(f"\nAnalyzing image: {image_name}")
            
            results = predict_image(model, image_path)
            if results:
                print("\nResults:")
                print(f"Cancer Detection: {results['cancer_detection']}")
                print(f"Cancer Probability: {results['cancer_probability']}")
                print(f"Cancer Stage: {results['cancer_stage']}")
                print(f"Stage Probability: {results['stage_probability']}")
                print("-" * 50)

if __name__ == "__main__":
    main() 