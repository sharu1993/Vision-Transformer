import utils
import vision_transformer_model as vit
import dataset
import train_class
import torch
import torchvision.transforms as transforms
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature


if __name__=="__main__":
    #mlflow init
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Vision Transformer")
    print("Tracking URI: ",mlflow.get_tracking_uri())
    mlflow_run=mlflow.start_run()
    mlflow.set_tags({
        "model":"ViT",
        "dataset":"CIFAR10"
    })

    model_cfg, train_cfg, data_cfg=utils.read_config("model.config")

    mlflow.log_params(utils.flatten_dict(model_cfg))
    mlflow.log_params(utils.flatten_dict(train_cfg))
    mlflow.log_params(utils.flatten_dict(data_cfg))

    #create model
    model=vit.VisionTransformer(
        img_size=model_cfg['img_size'],
        patch_size=model_cfg['patch_size'],
        num_classes=data_cfg['num_classes'],
        embed_dim=model_cfg['embed_dim'],
        depth=model_cfg['depth'],
        num_heads=model_cfg['num_heads'],
        mlp_dim=model_cfg['mlp_dim'],
        dropout=model_cfg['dropout']
    )

    #create transforms to pass to data class
    transforms_train=transforms.Compose([
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    transforms_valid=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    #class for handling data
    data = dataset.Dataset(
        data_cfg['dataset'],
        is_predefined_data=data_cfg['is_predefined_data'],
        data_path=data_cfg['root'],
        batch_size=train_cfg['batch_size'],
        valid_split=data_cfg['valid_split'],
        transforms_train=transforms_train,
        transforms_valid=transforms_valid,
        transforms_test=transforms_valid
        )
    data.Initialize()

    
    #criterion
    criterion=torch.nn.CrossEntropyLoss(label_smoothing=data_cfg['label_smoothing'])
    #optimizer
    optimizer=torch.optim.AdamW(model.parameters(),
                                lr=train_cfg['learning_rate'],
                                weight_decay=train_cfg['weight_decay'],
                                )
    #scheduler
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=train_cfg['schedule_factor'],
        patience=train_cfg['patience']
    )
    mlflow.log_dict({
        "Optimizer":"AdamW",
        "Criterion":"CrossEntropy",
        "Scheduler":"LROnPlateau"
    },"train_config.json")

    #training class
    trainbox=train_class.TrainBox(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        data=data,
        epochs=train_cfg['epochs']
    )

    #start training
    trainbox.run()

    #evaluate
    trainbox.eval()

    mlflow.end_run()

