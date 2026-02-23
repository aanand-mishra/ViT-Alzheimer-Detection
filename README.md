# Alzheimer's Disease Detection using Vision Transformer (ViT)

Hey! This is my project where I tried to use a Vision Transformer model to classify brain MRI scans into different stages of Alzheimer's disease. I'm still learning so bear with me lol.

## What does it do?

It takes brain MRI images and classifies them into one of 4 categories:
- Non Demented
- Very Mild Dementia
- Mild Dementia
- Moderate Dementia

## How it works

I used a pretrained ViT (Vision Transformer) model from Google (`google/vit-base-patch16-224`) and fine-tuned it on an Alzheimer's MRI dataset. Basically the idea is that instead of using a regular CNN, ViT splits the image into patches and uses attention mechanisms to classify it. I honestly still don't fully understand all of it but it works really well!

The training was done in two stages:
1. First I only trained the classification head (the last layer) for 10 epochs
2. Then I unfroze all the layers and fine-tuned the whole model for 20 more epochs with a smaller learning rate

## Results

The model got **99.94% accuracy** on the test set which honestly surprised me, I was not expecting it to do that well.

## Requirements

```
torch
torchvision
transformers
tqdm
Pillow
matplotlib
```

You can install them with:
```
pip install torch torchvision transformers tqdm Pillow matplotlib
```

## Dataset

I used an Alzheimer's MRI dataset. The images were already preprocessed/transformed. The dataset was split into train and test folders, each with subfolders for the 4 classes.

## How to run

### Training
Just run the notebook cells in order. Make sure to update the dataset paths to wherever you have the data stored.

```python
train_dataset = datasets.ImageFolder('your/path/to/TransformedTrain', transform=train_transform)
test_dataset = datasets.ImageFolder('your/path/to/TransformedTest', transform=test_transform)
```

### Predicting on a new image
```python
image_path = 'path/to/your/image.jpg'
predicted_class = predict_image(image_path, model, transform)
print(f'Predicted Class: {predicted_class}')
```

### Loading the saved model
The trained model weights are saved as `ViT_Alzheimer.pth`. To load it:

```python
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.classifier = nn.Linear(model.config.hidden_size, 4)
model.load_state_dict(torch.load('ViT_Alzheimer.pth'))
model.eval()
```

## Notes / things I learned

- Training only the head first before fine-tuning the whole model really helped. The loss dropped a LOT once I unfroze all the layers (went from ~0.42 down to like 0.004)
- I used AdamW optimizer which apparently works better than regular Adam for transformers
- Data augmentation (random rotations and flips) was added for the training set
- GPU is pretty much required for this, it would take forever on CPU

## Things I want to improve later

- Add a confusion matrix to see which classes the model struggles with
- Try with other ViT variants (larger models maybe?)
- Build a simple web app where you can upload your own image
- Add proper validation set during training instead of just train/test

## References

- [HuggingFace ViT](https://huggingface.co/google/vit-base-patch16-224)
- [Alzheimer's Dataset on Kaggle](https://www.kaggle.com/) (I don't remember the exact link sorry)
- [Attention Is All You Need paper](https://arxiv.org/abs/1706.03762) - original transformer paper
