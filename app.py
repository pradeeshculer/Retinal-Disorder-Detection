# Import necessary libraries
from flask import Flask, render_template, request, jsonify
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

# Create a Flask web application
app = Flask(__name__)

# Load your pre-trained deep learning model
# Specify the full path to your H5 file
model_path = r'E:\Documents\Docs\Pradeesh\Academic\VESIT\CNN_project\webpage\Lord_cnn.h5'

# Load the model
model = tf.keras.models.load_model(model_path)
categories = ["CNV", "DME", "DRUSEN", "NORMAL"]

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image from the request
        uploaded_file = request.files['file']
        
        # Save the uploaded image to a temporary file
        image_path = 'temp_image.jpg'
        uploaded_file.save(image_path)
        
        # Prepare the image for model prediction
        image = load_img(image_path, target_size=(128, 128))
        img_result = img_to_array(image)
        img_result = np.expand_dims(img_result, axis=0)
        img_result = img_result / 255.0
        
        # Make predictions using the loaded model
        predictions = model.predict(img_result)

        # Get the predicted class index
        predicted_index = np.argmax(predictions)
        predicted_category = categories[predicted_index]

        # Return the prediction as a JSON response
        return jsonify({'prediction': predicted_category})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the web application
if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')