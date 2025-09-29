# Pest-and-Disease-Detection-for-Plants
This project implements a Convolutional Neural Network (CNN) for detecting pests and diseases in plant images. The model is trained to classify images into different categories of plant diseases and pest infestations.


# Pest and Disease Detection in Plants Using CNN

This project implements a Convolutional Neural Network (CNN) for detecting pests and diseases in plants using image classification. The model is trained to identify different types of plant conditions using TensorFlow and Keras.

## Project Overview

The system uses a deep learning approach to classify plant images into different categories of pests and diseases. It employs a CNN architecture with multiple convolutional layers, max-pooling layers, and dense layers for effective feature extraction and classification.

## Model Architecture

The CNN model consists of:
- Input layer accepting images of size 150x150x3
- 3 Convolutional layers with ReLU activation
- MaxPooling layers for dimension reduction
- Flatten layer to convert 3D features to 1D
- Dense layer with 128 units and ReLU activation
- Dropout layer (0.5) for regularization
- Output layer with softmax activation for 4 classes

## Features

- **Data Augmentation**: Implements image augmentation techniques including:
  - Rotation
  - Width/Height shifts
  - Shear transformation
  - Zoom
  - Horizontal flips
  
- **Model Training**: 
  - Uses Adam optimizer
  - Categorical crossentropy loss function
  - Accuracy metrics tracking
  - Training history visualization

- **Evaluation Tools**:
  - Confusion matrix visualization
  - Test set accuracy measurement
  - Individual image prediction capability
  - Multi-image prediction display

## Project Structure

```
├── image data/
│   ├── train/
│   ├── validation/
│   └── test/
├── pest_detection_model.h5    # Saved trained model
├── training_history.pkl       # Training history data
└── detect.ipynb              # Main notebook with model implementation
```

## Usage

1. **Model Training**:
   ```python
   # The model is trained using:
   history = model.fit(
       train_generator,
       steps_per_epoch=steps_per_epoch,
       epochs=10,
       validation_data=validation_generator,
       validation_steps=validation_steps
   )
   ```

2. **Making Predictions**:
   ```python
   # For single image prediction:
   predict_image(image_path)

   # For multiple random predictions:
   display_predictions(num_images=5)
   ```

3. **Visualizing Results**:
   - Training/validation accuracy plots
   - Confusion matrix
   - Individual prediction results with confidence scores

## Model Performance

The model's performance can be evaluated using:
- Test accuracy metrics
- Confusion matrix visualization
- Real-time prediction confidence scores

## Dependencies

- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn
- Pickle

## Dataset Organization

The dataset should be organized in the following structure:
```
image data/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── validation/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

## Future Improvements

1. Implement additional data augmentation techniques
2. Experiment with different model architectures
3. Add support for real-time detection
4. Integrate transfer learning with pre-trained models
5. Implement model quantization for faster inference

