from os import path
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import cv2 as cv


class CNN(nn.Module):
    def __init__(self):
       super(CNN,self).__init__()
       #1,28,28
       self.conv1 = nn.Sequential(
               nn.Conv2d(1,8,kernel_size=5,stride=1,padding=2),
               nn.ReLU(),
               nn.MaxPool2d(2)
        )

       #8,14,14
       self.conv2 = nn.Sequential(
               nn.Conv2d(8,32,kernel_size=5,stride=1,padding=2),
               nn.ReLU(),
               nn.MaxPool2d(2)
        )

       #32,7,7
       self.out = nn.Linear(7*7*32,10)

    def forward(self,x):
       x = self.conv1(x)
       x = self.conv2(x)
       x = x.view(x.size(0),-1)
       return self.out(x)

BATCH_SIZE =50

def check_dataset():
    has_download = path.exists('./mnist')
    train_data = torchvision.datasets.MNIST(
        root ='./mnist',    
        train = True, 
        transform = torchvision.transforms.ToTensor(),
        download = not has_download)
    TRAIN_LOADER = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    return TRAIN_LOADER

def main(train_loader):
    nn = CNN()
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss()
    print(nn)
    for i in range(2):
        for step,(x,y) in enumerate(train_loader):
            print(x)
            print(y)
            results = nn(x)
            loss = loss_func(results,y)
            print("loss -> ",loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pass
    torch.save(nn, "./model")
    return nn


def test(nn):
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
    test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:10]
    print(test_x.shape)
    test_y = test_data.targets[:10]
    test_output = nn(test_x)
    pred_y = torch.max(test_output, 1)[1].numpy()
    print(pred_y, 'prediction number')
    print(test_y.numpy(), 'real number')
    img = cv.imread("./test_2.png",cv.IMREAD_GRAYSCALE)
    p = cv.resize(img,(28,28))
    p = cv.fastNlMeansDenoising(p)
    _, p = cv.threshold(p, 127, 255, cv.THRESH_BINARY_INV)
    #cv.imshow("src", p) 
    #cv.waitKey(0) 
    input_img = torch.from_numpy(np.array(p))
    input_img = torch.unsqueeze(input_img,0).type(torch.FloatTensor)
    input_img = torch.unsqueeze(input_img,0)
    print(input_img.shape)
    out_png = nn(input_img)
    print(torch.max(out_png,1)[1].numpy())

if __name__ == '__main__':
    if path.exists("./model"):
        model = torch.load("./model")
        test(model)
    else:
        loader =  check_dataset()
        nn = main(loader)
        test(nn)
