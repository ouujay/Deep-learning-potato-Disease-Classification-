# Potato Disease Classification using Deep Learning

A deep learning project that uses Convolutional Neural Networks (CNN) to classify potato leaf diseases. The model can identify three classes: Early Blight, Late Blight, and Healthy potato leaves.

## Overview

This project implements an image classification model using TensorFlow/Keras to help farmers identify common potato diseases from leaf images. Early detection of diseases like Early Blight and Late Blight can help prevent crop losses and improve agricultural yield.

## Dataset

The model is trained on the [PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) available on Kaggle.

- **Total Images**: 2,152
- **Classes**:
  - `Potato___Early_blight`
  - `Potato___Late_blight`
  - `Potato___healthy`
- **Image Size**: 256 x 256 pixels

### Data Split
| Set | Percentage | Batches |
|-----|------------|---------|
| Training | 80% | 54 |
| Validation | 10% | 6 |
| Testing | 10% | 8 |

## Model Architecture

The model uses a Sequential CNN architecture with the following layers:

```
Input (256, 256, 3)
    ↓
Resize & Rescale (Normalization)
    ↓
Conv2D (32 filters, 3x3) + ReLU → MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU → MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU → MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU → MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU → MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU → MaxPooling2D (2x2)
    ↓
Flatten → Dense (64, ReLU) → Dense (3, Softmax)
```

**Total Parameters**: 183,747 (717.76 KB)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 50 |
| Batch Size | 32 |
| Image Size | 256 x 256 |

## Results

The model achieved excellent performance on the test set:

- **Test Accuracy**: 100%
- **Test Loss**: 0.0017

## Project Structure

```
├── potato disease classification.ipynb  # Main Jupyter notebook
├── potato_model.h5                       # Saved model (HDF5 format)
├── potato_model.keras                    # Saved model (Keras format)
├── LICENSE                               # MIT License
└── README.md                             # Project documentation
```

## Requirements

- Python 3.x
- TensorFlow 2.x
- Matplotlib
- NumPy

Install dependencies:
```bash
pip install tensorflow matplotlib numpy
```

## Usage

### Training

1. Download the [PlantVillage dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) from Kaggle
2. Open `potato disease classification.ipynb` in Jupyter Notebook or Google Colab
3. Update the dataset path in the notebook
4. Run all cells to train the model

### Inference

```python
import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model('potato_model.keras')

# Class names
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Load and preprocess image
img = tf.keras.preprocessing.image.load_img('path/to/image.jpg', target_size=(256, 256))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# Predict
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions[0])]
confidence = round(100 * np.max(predictions[0]), 2)

print(f"Predicted: {predicted_class} with {confidence}% confidence")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Oluwafunbi Onaeko

## Acknowledgments

- [PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) for providing the training data
- TensorFlow/Keras team for the deep learning framework
