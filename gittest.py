import torch
from torch import nn
import torch.nn.functional as F
import torchvision 
import matplotlib.pyplot as plt
from statistics import mean

class FFNN(nn.Module):
    def __init__(self):
        #defines layers
        super().__init__()
        output_dimensions  = 1
        input_dimensions = 1 
        self.linear_layer1 = nn.Linear(28*28, 392) #input pixel dimension is 28x28
        self.linear_layer2 = nn.Linear(392, 150) #output dimension of the prev layer has to equal 
        self.linear_layer3 = nn.Linear(150, 10) #input dimension of the next
        #self.linear_layer4 = nn.Linear(98, 49)
        #self.linear_layer5 = nn.Linear(49, 10)  #10 because of the digit probability 
    def forward(self, x): #x is a collection of items in a batch; multiple different images at the same time 
        #x is (batch, 28, 28), but linear layer expects (batch, 28 * 28). Step 1 is to transform (28, 28) to (28*28)
        #push through layers of the model 
        x = x.view(-1, 784)
        x = F.relu(self.linear_layer1(x))
        x = F.relu(self.linear_layer2(x))
        x = F.relu(self.linear_layer3(x))
        #x = F.relu(self.linear_layer4(x))
       # x = F.relu(self.linear_layer5(x))

        return x

model = FFNN()  

train_dataset = torchvision.datasets.MNIST(
    "files/",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=torchvision.transforms.Compose(
        [lambda x: torch.LongTensor([x]), lambda x: F.one_hot(x, 10)]
    ),
)
training_set_size = 600
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_set_size, shuffle=True)

test_dataset = torchvision.datasets.MNIST(
    "files/",
    train=False,
    download=False,
    transform=torchvision.transforms.ToTensor(),
    target_transform=torchvision.transforms.Compose(
        [lambda x: torch.LongTensor([x]), lambda x: F.one_hot(x, 10)]
    ),
)
testing_set_size = 600

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=testing_set_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()


#calculate average of losses and plot on a graph (pyplot) to see how the model is getting better over time
#return training epoch (100) accuracy and also the number run 
def train_epoch(model, optimizer, dataloader):
    model.train() # You have to tell your model to go into "train" mode
    losses = []
    for input, labels in dataloader:
        #input, labels = input.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, torch.squeeze(labels).float())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return losses # Typically you want to keep a list of all the batch losses


def test_epoch(model, dataloader):
    model.eval()
    losses = []
    
    with torch.no_grad():
        for input, labels in dataloader:
           # input, labels = input.to(DEVICE), labels.to(DEVICE)
            output = model(input)
            loss = criterion(output, torch.squeeze(labels).float())
            losses.append(loss.item())
        #print(losses)
    return losses

training = []
validation = []
epoch = 10
epoch_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i in range(epoch):
    
    training.append(mean(train_epoch(model, optimizer, train_loader)))
    validation.append(mean(test_epoch(model, test_loader)))

plt.plot(epoch_list, training, color = "blue")
plt.plot(epoch_list, validation, color = "red")

plt.show()

