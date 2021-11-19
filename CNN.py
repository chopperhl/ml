import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
       super(CNN,self).__init__()
       #1,16,16
       self.conv1 = nn.Sequential(
               nn.Conv2d(1,2,kernel_size=5,stride=1,padding=2),
               nn.ReLU(),
               nn.MaxPool2d(2)
        )

       #2,8,8
       self.conv2 = nn.Sequential(
               nn.Conv2d(2,4,kernel_size=5,stride=1,padding=2),
               nn.ReLU(),
               nn.MaxPool2d(2)
        )

       #4,4,4
       self.out = nn.Linear(4*4*4,1)

    def forward(self,x):
       x = self.conv1(x)
       x = self.conv2(x)
       x = x.view(x.size(0),-1)
       return self.out(x)

def main():
    nn = CNN()
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.1)
    loss_func = torch.nn.MSELoss()
    print(nn)
    for i in range(10000):
        input = torch.rand(1,1,16,16)
        max = input.max().view(1,1)
        results = nn(input)
        loss = loss_func(results,max)
        print(max)
        print(results)
        print("loss -> ",loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()


