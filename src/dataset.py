import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image

class SplitImages():

    def __init__(self):
        #Split the dataset into train and test folder 
        #Get all image paths using glob
        self.data_paths = glob("../Animals/*/*")  
        labels = [path.split("/")[-2] for path in self.data_paths]  
        self.train_paths, self.test_paths = train_test_split(self.data_paths, test_size=0.2, stratify=labels)

        self.copy_files(self.train_paths, 'train')
        self.copy_files(self.test_paths, 'test')

    def copy_files(self, file_list, destination):
        for file_path in file_list:
            dest_path = file_path.replace('Animals', destination, 1)  
            os.makedirs(dest_path.rsplit("/", 1)[0], exist_ok=True)  
            shutil.copy(file_path, dest_path)  


class CustomDataset(Dataset):
    def __init__(self, img_paths, transform=None, class_mapping=None):
        self.img_paths=img_paths
        self.transform=transform
        self.class_mapping = class_mapping or {}

    def __len__(self):
        return len(self.img_paths)
 
    def __getitem__(self, index):
        image_path=self.img_paths[index]
        image=Image.open(image_path).convert("RGB")

        if self.transform:
            img=self.transform(image)
        
        label_name=image_path.split("/")[-2]
        label = self.class_mapping[label_name]

        return img, label


if __name__ =='__main__':
    splitter=SplitImages()

    class_name=[d.split("/")[2] for d in glob("../train/*")]
    train_paths=glob("../train/*/*")
    test_paths=glob("../test/*/*")

    print(class_name)
    print(len(splitter.data_paths))
    print(len(train_paths))
    print(len(test_paths))

#-----------------------------------------
    
