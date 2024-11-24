from torch.utils.data import Dataset
from PIL import Image
import os
import random

class MyDataset(Dataset):
    
    def __init__(self, data_dir, transform=None, reshape = False):
        self.data_dir = data_dir
        self.label_name = {}
        self.data_info = self.get_img_info()
        random.seed(0)
        random.shuffle(self.data_info)
        self.transform = transform
        self.reshape = reshape

        
    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     
 
        if self.reshape:
            x = 15
            y = 0 
            width = 64
            height = 64
            (l,h) = img.size
            rate = min(l,h) / width
            img = img.resize((int(l // rate),int(h // rate)),Image.BILINEAR) 
            img = img.crop((x,y,width+x,height+y))

        if self.transform is not None:
            img = self.transform(img)
 
        return img, label

    
    def __len__(self):
        return len(self.data_info)
    
    
    def get_img_info(self):
        data_info = list()
        for root, dirs, _ in os.walk(self.data_dir):
            # 遍历类别
            for idx in range(len(dirs)):
                sub_dir = dirs[idx]
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                self.label_name[sub_dir] = idx
                
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = idx
                    data_info.append((path_img, label))
        return data_info