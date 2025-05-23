import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

def get_pytorch_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transforms, test_transforms

def get_pytorch_dataloaders(data_dir='dataset', batch_size=32):
    train_transforms, test_transforms = get_pytorch_transforms()
    train_dataset = datasets.ImageFolder(root=f'{data_dir}/training', transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=f'{data_dir}/testing', transform=test_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset.classes

def get_tensorflow_generators(data_dir='dataset', batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        f'{data_dir}/training',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse'
    )
    test_generator = test_datagen.flow_from_directory(
        f'{data_dir}/testing',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse'
    )
    return train_generator, test_generator, train_generator.class_indices