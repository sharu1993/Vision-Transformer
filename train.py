import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import vision_transformer_model as vit
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import utils
import pdb

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

config=utils.read_config("model.config")

transform=transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
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
train_size = int((1-config['data']['valid_split'])*len(trainset_full))
val_size=len(trainset_full)-train_size

trainset,valset = torch.utils.data.random_split(
    trainset_full,
    [train_size,val_size],
    generator=torch.Generator().manual_seed(15) #have a fixed seed for now
)
valset.dataset.transform=transform

trainloader=torch.utils.data.DataLoader(
    trainset,
    batch_size=config['training']['batch_size'],
    shuffle=True
)

valloader=torch.utils.data.DataLoader(
    valset,
    batch_size=config['training']['batch_size'],
    shuffle=False
)

testloader=torch.utils.data.DataLoader(
    testset,
    batch_size=config['training']['batch_size'],
    shuffle=False
)



model=vit.VisionTransformer(
    img_size=config['model']['img_size'],
    patch_size=config['model']['patch_size'],
    embed_dim=config['model']['embed_dim'],
    depth=config['model']['depth'],
    num_heads=config['model']['num_heads'],
    mlp_dim=config['model']['mlp_dim'],
    num_classes=config['data']['num_classes']
).to(device)

criterion=nn.CrossEntropyLoss(label_smoothing=config['data']['label_smoothing'])
optimizer=optim.AdamW(model.parameters(),lr=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'])
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,mode='min',factor=0.01,patience=30
)
epochs=config['training']['epochs']

#log dicts
train_config={
    "optimizer":"AdamW",
    "Criterion":"Cosine",
    "Scheduler":"LROnPLateau"
}

mlflow.log_dict(train_config,"train_config.json")
#log hyperparameters
mlflow.log_params(utils.flatten_dict(config))


#training loop
for epoch in range(epochs):
    model.train()
    train_loss=0
    cur_lr=optimizer.param_groups[0]['lr']
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
    best_val_loss=float('inf')
    
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

    scheduler.step(val_loss)
    
    print(
        f"Epoch {epoch}, Train Loss: {train_loss/len(trainloader):.4f}, Validation Loss: {val_loss:.4f}, Valid Acc. : {100*accuracy:.4f}"
    )
    mlflow.log_metric("train_loss",train_loss/len(trainloader),step=epoch+1)
    mlflow.log_metric("valid_loss",val_loss,step=epoch+1)
    mlflow.log_metric("valid_acc",accuracy,step=epoch+1)
    mlflow.log_metric("cur_lr",cur_lr,step=epoch+1)
    if best_val_loss>val_loss:
        #save model with best validation loss
        torch.save({
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "val_acc":accuracy,
            "val_loss":val_loss,
            "scheduler_state_dict":scheduler.state_dict()
        },"best_vit_model.pt")
        print(f"Model saved")


mlflow.log_metric("Best Validation Accuracy",best_val_loss)

#evaluation (load best model)
saved_checkpoint=torch.load('best_vit_model.pt')
eval_model=vit.VisionTransformer(
    img_size=config['model']['img_size'],
    patch_size=config['model']['patch_size'],
    embed_dim=config['model']['embed_dim'],
    depth=config['model']['depth'],
    num_heads=config['model']['num_heads'],
    mlp_dim=config['model']['mlp_dim'],
    num_classes=config['data']['num_classes']
)
eval_model.load_state_dict(saved_checkpoint['model_state_dict'])
eval_model.eval()
model.to(device)
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