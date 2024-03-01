from flask import Flask, request, jsonify
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

# 加载模型
model = models.resnet152(pretrained=False)
model.load_state_dict(torch.load('resnet152_weights.pth', map_location=torch.device('cpu')))
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 类别标签
LABELS = ['label_1', 'label_2', 'label_3']  # 替换为您的类别标签

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']
    img = Image.open(image).convert('RGB')
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        label = LABELS[predicted.item()]

    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
