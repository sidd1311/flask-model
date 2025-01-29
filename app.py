import torch
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import requests
from torchvision import transforms

# Initialize the Flask app
app = Flask(__name__)

# Load the model globally to avoid reloading it on each request
best_model = torch.load('model.pth')
best_model.eval()

# If CUDA is available, move the model to the GPU
if torch.cuda.is_available():
    best_model = best_model.cuda()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Prediction function
def predict1(x):
    if x.startswith("http"):
        response = requests.get(x, stream=True)
        response.raise_for_status()
        img = Image.open(response.raw).convert("RGB")
    else:
        img = Image.open(x).convert("RGB")
    
    img = transform(img).unsqueeze(0)  # Add batch dimension
    
    if torch.cuda.is_available():
        img = img.cuda()
    
    with torch.no_grad():
        out = best_model(img)
    
    return out.argmax(1).item()

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_url = data['url']

    try:
        predicted_class = predict1(image_url)
        return jsonify({'predicted_class': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
