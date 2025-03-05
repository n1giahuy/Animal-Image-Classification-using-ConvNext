import yaml
from torch.utils.data import DataLoader, Dataset
from glob import glob
import shutil
import torch.nn as nn
import torch.optim as optim
import torch
from src.dataset import CustomDataset
from src.models import ConvNextClassifier, get_transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as file :
        config=yaml.safe_load(file)
    return config

config=load_config('config.yaml')

train_paths = glob(f"{config['train-root']}/*/*")
test_paths = glob(f"{config['test-root']}/*/*")

# print(len(train_paths))
# print(len(test_paths))

class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
class_mapping = {class_name: idx for idx, class_name in enumerate(class_names)}
    

train_transforms=get_transforms(img_size=config['image-size'], is_training=True) 
test_transforms=get_transforms(img_size=config['image-size'], is_training=False) 

train_dataset=CustomDataset(train_paths, train_transforms, class_mapping=class_mapping)
test_dataset=CustomDataset(test_paths, test_transforms, class_mapping=class_mapping)

train_loader=DataLoader(train_dataset, batch_size=config['batch-size'], shuffle=True, num_workers=1, drop_last=True )
test_loader=DataLoader(test_dataset, batch_size=config['batch-size'], shuffle=False, num_workers=1, drop_last=False )

def plot_confussion_matrix(cm, class_names):
    fig, ax= plt.subplots(figsize=(10,8))
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig


if __name__ =='__main__':
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=ConvNextClassifier(num_classes=config['num-classes'], model_size='base', pretrained=True, freeze_backborn=True).to(device)
    
    input_data=torch.rand(8,3,224,224).to(device)
    output=model(input_data) 
    
    print(output.size())
    criterion=nn.CrossEntropyLoss()
    
    optimizer=optim.Adam(model.parameters(), lr=config['learning-rate'])

    num_epochs=config['epochs']
    writer=SummaryWriter(config['logging-dir'])

    def train(epoch):
        model.train()
        running_loss=0.0
        total, correct=0, 0

        progress_bar=tqdm(train_loader, desc='Training', colour='green')
        for iter, (images, labels) in enumerate(progress_bar):
            images, labels=images.to(device), labels.to(device)

            #Forward
            outputs=model(images)
            loss=criterion(outputs, labels)

            #*Train Accuracy
            predicted=torch.argmax(outputs, dim=1)
            correct +=(predicted==labels).sum().item()
            total+=labels.size(0)
            accuracy_score = correct / total

            progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} Iteration {iter+1}/{len(train_loader)} Accuracy {accuracy_score:.4f} Loss {loss:.4f}")
            writer.add_scalar('Train/Loss', loss, epoch*len(train_loader)+iter)
            writer.add_scalar('Train/Accuracy', accuracy_score, epoch*len(train_loader)+iter)

            
            #Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   
            running_loss += loss.item()
        
    
    def test(epoch):
        model.eval()
        total, correct= 0, 0
        running_loss=0.0
        all_preds=[]
        all_labels=[]

        with torch.no_grad(): #disable gradient tracking because we don't need to update the model  
            progress_bar=tqdm(test_loader, desc="testing", colour='red')
            for iter, (images, labels) in enumerate(progress_bar):
                images, labels=images.to(device), labels.to(device)
                predictions=model(images)
                loss=criterion(predictions, labels)
                running_loss+= loss.item()
                
                #*Calculate accuracy 
                predicted=torch.argmax(predictions, dim=1)
                correct+= (predicted==labels).sum().item()
                total+=labels.size(0) #Get the batch size as an integer 
                accuracy_score = correct / total

                #*Store Predictions&Labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} Iteration {iter+1}/{len(test_loader)} Accuracy {accuracy_score:.4f} Loss {loss:.4f}")
                writer.add_scalar('Test/Loss', loss, epoch*len(test_loader)+iter)
                writer.add_scalar('Test/Accuracy', accuracy_score, epoch*len(test_loader)+iter)
            
        cm=confusion_matrix(all_labels, all_preds)
        
        writer.add_figure('Test/Confusion Matrix',plot_confussion_matrix(cm, class_names), global_step=epoch)
        return accuracy_score
    
    def load_checkpoint(path):
        if shutil.os.path.isfile(path):
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            start_epoch = checkpoint.get("epoch", 0)
            best_acc = checkpoint.get("best_acc", 0.0)
        else:
            start_epoch = 0
            best_acc = 0.0
        return start_epoch, best_acc

    start_epoch, best_acc = load_checkpoint("model_save/last_save.pt")

    for epoch in range(start_epoch,num_epochs):
        train(epoch)
        test_acc=test(epoch)

        last_checkpoint = {
            "epoch": epoch+1,
            "best_acc": best_acc,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(last_checkpoint, "model_save/last_save.pt")

        if test_acc > best_acc:
            best_acc=test_acc
            best_checkpoint = {
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(best_checkpoint, "model_save/best_model.pt")

