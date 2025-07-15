# Spinach-Mobilenetv3
# Spinach Disease Detection

A deep learning-based system for detecting diseases in spinach leaves using MobileNetV3 and TensorFlow Lite.

## Project Structure

```
SpinachDiseaseDetection/
├── data/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
├── models/
│   ├── mobilenetv3_finetuned.h5    # Trained Keras model
│   ├── spinach_disease_model.tflite # TFLite model for Android
│   ├── training_history.png        # Training history plot
│   └── confusion_matrix.png        # Model evaluation results
├── notebooks/          # Jupyter notebooks for analysis
├── scripts/
│   ├── train_model.py  # Model training script
│   └── evaluate_model.py # Model evaluation script
├── android_app/
│   └── app_code/       # Android application code
└── requirements.txt    # Python dependencies
```

## Setup Instructions

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare the dataset:
   - Organize your images in the following structure:
     ```
     Malabar_Dataset/
     ├── Healthy-Leaf/
     ├── Downy-Mildew/
     ├── Bacterial-Spot/
     ├── Anthracnose/
     └── Pest-Damage/
     ```
   - Each class folder should contain the corresponding images

4. Train the model:
```bash
python scripts/train_model.py
```
This will:
- Train the MobileNetV3 model
- Save the trained model
- Convert it to TFLite format
- Generate training history plots

5. Evaluate the model:
```bash
python scripts/evaluate_model.py
```
This will:
- Generate predictions on the test set
- Create a confusion matrix
- Print classification metrics

## Model Architecture

- Base Model: MobileNetV3Large (pre-trained on ImageNet)
- Input Size: 224x224 RGB images
- Output Classes: 5 (Healthy, Downy Mildew, Bacterial Spot, Anthracnose, Pest Damage)
- Additional Layers:
  - Global Average Pooling
  - Dense (1024 units, ReLU)
  - Dropout (0.5)
  - Dense (512 units, ReLU)
  - Dropout (0.3)
  - Dense (5 units, Softmax)

## Training Details

- Data Augmentation:
  - Rotation: ±20 degrees
  - Width/Height Shift: ±20%
  - Shear: ±20%
  - Zoom: ±20%
  - Horizontal Flip
  - Rescaling: 1/255

- Training Strategy:
  - Two-phase training:
    1. Train top layers (base model frozen)
    2. Fine-tune last 30 layers of base model
  - Early stopping with patience=5
  - Learning rate reduction on plateau
  - Batch size: 32
  - Initial epochs: 50
  - Fine-tuning epochs: 20

## Performance Metrics

- Training history plots are saved in `models/training_history.png`
- Confusion matrix visualization in `models/confusion_matrix.png`
- Detailed classification report printed during evaluation

## Android App Integration

1. Copy the TFLite model to the Android app:
```bash
cp models/spinach_disease_model.tflite android_app/app_code/app/src/main/assets/
```

2. Build the Android app:
```bash
cd android_app/app_code
./gradlew assembleDebug
```

## Troubleshooting

1. Memory Issues:
   - Reduce batch size in `train_model.py`
   - Use smaller image size
   - Enable memory growth for GPU

2. Overfitting:
   - Increase dropout rates
   - Add more data augmentation
   - Reduce model complexity

3. TFLite Conversion Issues:
   - Ensure TensorFlow version compatibility
   - Check model architecture for unsupported operations
   - Verify input/output tensor shapes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: Malabar Spinach Disease Dataset
- Base Model: MobileNetV3 (Google Research)
- TensorFlow and Keras teams 
