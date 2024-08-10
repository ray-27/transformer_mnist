import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from mnist_dataset import mnist_dataset,mnist_test
from transformer_model import vision_transform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from colorama import init, Fore
init(autoreset=True)
import config

def device_is():
    if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Device is " + Fore.GREEN + "CUDA")
    else:
        device = torch.device('cpu')
        print("Device is CPU")
    
    return device


# import data
def train():
    data = pd.read_csv('data/train.csv')

    train_data, val_data = train_test_split(data,test_size=0.1,shuffle=True)

    train_dataset = mnist_dataset(train_data)
    val_dataset = mnist_dataset(val_data)

    train_loader = DataLoader(train_dataset,batch_size=config.batch_size,num_workers=config.num_workers,shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_dataset,batch_size=config.batch_size,num_workers=config.num_workers,shuffle=True,pin_memory=True)

    # load the model and optimizer

    device = device_is()
    
    model = vision_transform(emb_dim=config.emb_dim,
                             num_of_heads=config.num_of_heads,
                             patch_size=config.patch_size,
                             dropout_rate=config.dropout_rate,
                             num_of_encoder_blocks=config.num_of_encoder_blocks,
                             num_classes=config.num_classes,
                             show_params=config.show_params).to(device)
    
    optimizer = Adam(model.parameters(),lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss() 
    epoch_loss_dir = []
    val_loss_dir = []
    accuracy_dir = []

    # training loop 
    for epoch in tqdm(range(config.epoches)):
        model.train()
        running_loss = 0

        for X,label in train_loader:
            X,label = X.to(device),label.to(device)

            output = model(X)
            loss = criterion(output,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_loss_dir.append(epoch_loss)
        tqdm.write(f'Epoch [{epoch+1}/{config.epoches}], Loss: {epoch_loss:.4f}')
    
        # Validation of the model
        model.eval()  # Set model to evaluation mode
        val_running_loss = 0.0
        correct_predictions = 0  # Counter for correct predictions
        total_predictions = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = correct_predictions / total_predictions * 100
        val_loss_dir.append(val_epoch_loss)
        accuracy_dir.append(val_accuracy)
        tqdm.write(f'Validation Loss: {val_epoch_loss:.4f}, Model Accuracy : {val_accuracy:.4f}')

    print(Fore.GREEN + "Training Complete :)")

    # saving the model and plots
    print("Saving the model")
    torch.save(model,'down/mnist_model.pth')
    save_plots(epoch_loss_dir,val_loss_dir,accuracy_dir,config.epoches)

def test():
    test_data = pd.read_csv('data/test.csv')
    test_dataset = mnist_test(test_data)
    test_loader = DataLoader(test_dataset,batch_size=1,num_workers=1)

    device = device_is()

    model = torch.load('down/mnist_model.pth').to(device)
    predictions = []
    print(Fore.LIGHTMAGENTA_EX + "Makeing the predections")
    with torch.no_grad():
        for input in tqdm(test_loader):
            input = input.to(device)
            output = model(input)

            probs = torch.softmax(output,dim=1).to(device)
            predicted_classes = torch.argmax(probs,dim=1).to(device)

            predictions.append(predicted_classes.cpu().numpy())
    print(Fore.LIGHTYELLOW_EX + "predictions done :)")
    
    df_predictions = pd.DataFrame(predictions, columns=['Label'])
    df_predictions.index = pd.RangeIndex(start=1, stop=len(df_predictions)+1, step=1)
    df_predictions.to_csv('down/submission.csv', index=True,index_label='ImageId')

    print(Fore.LIGHTGREEN_EX + "submission.csv file created")
    
def save_plots(epoch_loss_dic,val_loss_dic,accuracy_dic,epoches):
    epo = list(range(1,epoches+1))
    plt.figure(figsize=(10, 5))  # Set the figure size (optional)
    plt.plot(epo,epoch_loss_dic, marker='o', linestyle='-', color='b', label='Loss per Epoch')
    plt.title('Loss vs. Epochs')  # Set the title of the graph
    plt.xlabel('Epoch')  # Label for the x-axis
    plt.ylabel('Loss')  # Label for the y-axis
    plt.grid(True)  # Enable grid for easier readability
    plt.xticks(epo)  # Set x-axis ticks to be exactly at the epochs numbers
    plt.legend()  # Show legend to identify the plot
    plt.savefig('down/loss_vs_epochs.png', format='png', dpi=300)

    plt.figure(figsize=(10, 5))  # Set the figure size (optional)
    plt.plot(epo,val_loss_dic, marker='o', linestyle='-', color='b', label='Loss per Epoch')
    plt.title('Validation vs. Epochs')  # Set the title of the graph
    plt.xlabel('Epoch')  # Label for the x-axis
    plt.ylabel('Loss')  # Label for the y-axis
    plt.grid(True)  # Enable grid for easier readability
    plt.xticks(epo)  # Set x-axis ticks to be exactly at the epochs numbers
    plt.legend()  # Show legend to identify the plot
    plt.savefig('down/val_vs_epochs.png', format='png', dpi=300)

    plt.figure(figsize=(10, 5))  # Set the figure size (optional)
    plt.plot(epo,accuracy_dic, marker='o', linestyle='-', color='b', label='Loss per Epoch')
    plt.title('Accuracy vs. Epochs')  # Set the title of the graph
    plt.xlabel('Epoch')  # Label for the x-axis
    plt.ylabel('Loss')  # Label for the y-axis
    plt.grid(True)  # Enable grid for easier readability
    plt.xticks(epo)  # Set x-axis ticks to be exactly at the epochs numbers
    plt.legend()  # Show legend to identify the plot
    plt.savefig('down/accuracy_vs_epochs.png', format='png', dpi=300)


if __name__ == "__main__":
    # train()
    test()