# Dynamic Image Classifier using TensorFlow

This is a Streamlit application that uses a pre-trained TensorFlow model (MobileNetV2) to classify images uploaded by the user.

## Dependencies

- Streamlit
- TensorFlow
- PIL
- NumPy
- Matplotlib

## How it works

1. The application first loads the pre-trained MobileNetV2 model from TensorFlow's model zoo.

2. The user is prompted to upload an image file (JPG or PNG format).

3. The uploaded image is preprocessed to match the input requirements of the MobileNetV2 model. This involves resizing the image to 224x224 pixels and normalizing the pixel values.

4. The preprocessed image is then fed into the model to obtain predictions.

5. The model's predictions are decoded into human-readable class names and displayed on the screen, along with the confidence scores.

## Usage

To run the application, use the following command:

```bash
streamlit run main.py
```

Then, open a web browser and navigate to the URL displayed in the terminal. From the web interface, you can upload an image and see the model's predictions.