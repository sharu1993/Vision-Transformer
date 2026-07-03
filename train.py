import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import vision_transformer_model as vit
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature

#set up mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Vision Transformer")
print("Tracking URI: ",mlflow.get_tracking_uri())

mlflow_run=mlflow.start_run()

#set tags
mlflow.set_tags({
    "model":"ViT",
    "dataset":"CIFAR10"
})

if torch.backends.mps.is_available():
    device=torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    device=torch.diag_embed("cuda")
    print("Using CUDA")
else:
    device=torch.device("cpu")

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

trainset_full=torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

testset=torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

#create a validation set from the training data
train_size = int(0.9*len(trainset_full))
val_size=len(trainset_full)-train_size

trainset,valset = torch.utils.data.random_split(
    trainset_full,
    [train_size,val_size],
    generator=torch.Generator().manual_seed(15) #have a fixed seed for now
)
valset.dataset.transform=transform

trainloader=torch.utils.data.DataLoader(
    trainset,
    batch_size=128,
    shuffle=True
)

valloader=torch.utils.data.DataLoader(
    valset,
    batch_size=128,
    shuffle=False
)

testloader=torch.utils.data.DataLoader(
    testset,
    batch_size=128,
    shuffle=False
)



model=vit.VisionTransformer(
    img_size=32,
    patch_size=4,
    embed_dim=128,
    depth=6,
    num_heads=4,
    mlp_dim=256,
    num_classes=10
).to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
epochs=50

#log dicts
config={
    "optimizer":"AdamW",
    "Criterion":"Cosine"
}
mlflow.log_dict(config,"config.json")
#log hyperparameters
mlflow.log_params({
    "epochs":epochs,
    "batch_size":128,
    "patch_size":4,
    "depth":6,
    "heads":4,
    "learning_rate":3e-4
})


#training loop
for epoch in range(epochs):
    model.train()
    train_loss=0

    for images, labels in trainloader:
        images,labels=images.to(device),labels.to(device)

        optimizer.zero_grad()
        outputs=model(images)
        
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        train_loss+=loss
    #validation for each epoch
    model.eval()
    val_loss=0
    correct=0
    total=0
    
    with torch.no_grad():
        for images,labels in valloader:
            images,labels=images.to(device),labels.to(device)
            
            outputs=model(images)
            
            loss=criterion(outputs,labels)
            val_loss+=loss

            pred=outputs.argmax(dim=1)
            correct+=(pred==labels).sum().item()
            total+=labels.size(0)
        val_loss/=len(valloader)
        accuracy=correct/total

    print(
        f"Epoch {epoch}, Train Loss: {train_loss/len(trainloader):.4f}, Validation Loss: {val_loss:.4f}, Valid Acc. : {100*accuracy:.4f}"
    )
    mlflow.log_metric("train_loss",train_loss/len(trainloader),step=epoch+1)
    mlflow.log_metric("valid_loss",val_loss,step=epoch+1)
    mlflow.log_metric("valid_acc",accuracy,step=epoch+1)

#evaluation
model.eval()
correct=0
total=0

with torch.no_grad():
    for images,labels in testloader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        pred=outputs.argmax(dim=1)
        correct+=(pred==labels).sum().item()
        total+=labels.size(0)
    print(f"Test Accuracy: {100*correct/total:.2f}%")
    mlflow.log_metrics({
        "Accuracy":100*correct/total
    })

#save model later
mlflow.end_run()