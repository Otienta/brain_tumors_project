import argparse
import torch
import tensorflow as tf
from models.cnn import CustomCNN
from models.cnn_tf import create_cnn_model
from models.train import Trainer
from utils.prep import get_pytorch_dataloaders, get_tensorflow_generators

def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate brain tumor CNN models")
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--cuda', action='store_true')
    return parser.parse_args()

def train_pytorch_model(args, device):
    print("Training PyTorch model...")
    train_loader, test_loader, classes = get_pytorch_dataloaders(data_dir='dataset')
    model = CustomCNN(num_classes=len(classes)).to(device)
    trainer = Trainer(model, train_loader, test_loader, args.lr, args.wd, args.epochs, device)
    trainer.train(save_path='model.pth')
    trainer.evaluate()

def train_tensorflow_model(args):
    print("Training TensorFlow model...")
    train_generator, test_generator, class_indices = get_tensorflow_generators(data_dir='dataset')
    model = create_cnn_model(num_classes=len(class_indices))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, validation_data=test_generator, epochs=args.epochs, verbose=1)
    loss, accuracy = model.evaluate(test_generator, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}% | Test Loss: {loss:.4f}")
    model.save('model_tf.h5')
    print("Modèle TensorFlow sauvegardé sous model_tf.h5")

def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == 'train':
        # Entraîner les deux modèles
        args.epochs = 5  # Forcer 5 époques comme demandé
        train_pytorch_model(args, device)
        train_tensorflow_model(args)
    elif args.mode == 'eval':
        train_loader, test_loader, classes = get_pytorch_dataloaders(data_dir='dataset')
        model = CustomCNN(num_classes=len(classes)).to(device)
        try:
            model.load_state_dict(torch.load('model.pth', map_location=device))
            print("Loaded model.pth")
        except FileNotFoundError:
            print("Error: model.pth not found")
            return
        trainer = Trainer(model, train_loader, test_loader, args.lr, args.wd, args.epochs, device)
        trainer.evaluate()

if __name__ == '__main__':
    main()