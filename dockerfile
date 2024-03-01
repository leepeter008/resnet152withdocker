FROM pytorch/pytorch:latest

WORKDIR /app

COPY resnet152_model.py .
COPY resnet152_weights.pth .

RUN pip install flask pillow

EXPOSE 5000

CMD ["python", "resnet152_model.py"]
