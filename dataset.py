import torch
import torchvision
import torchvision.transforms as transforms
import utils
from typing import Optional, Tuple, Dict, Any
import pdb

class Dataset:
    #should allow for either using an existing dataset from the web
    #or use local data
    def __init__(
            self,
            dataset_name:str="CIFAR10", #dataset name. Make it workable with path to a data folder as well
            is_predefined_data:bool=True,  #true if using predefined datasets. False if using path
            data_path:str="./data",     #data directory to store/load or custom data path
            batch_size=128,             #batch size
            valid_split=0.1,            #percentage of training data to be used for validation
            transforms_train:Optional[transforms.Compose]=None, #Transforms for training
            transforms_test:Optional[transforms.Compose]=None,  #transforms for testing
            transforms_valid: Optional[transforms.Compose]=None,#transforms for validation
            num_workers:int=4,          #num of workers
            shuffle:bool=True,          #data shuffle ON/OFF
            pin_memory:bool=True,       #memory pinning
            **kwargs                    #extra arguments
            ):
        self.dataset_name=dataset_name
        self.batch_size=batch_size
        self.data_root=data_path
        self._valid_split=valid_split
        self._train_split=1.0-valid_split
        self.is_predefined_data=is_predefined_data
        self.shuffle=shuffle
        self.pin_memory=pin_memory
        self.num_workers=num_workers

        self.kwargs=kwargs

        #transforms
        self.transforms_train = transforms_train or self._get_default_transform()
        self.transforms_test = transforms_test or self._get_default_transform()
        self.transforms_valid = transforms_valid or self._get_default_transform()

        #datasets
        self.train_dataset=None
        self.valid_dataset=None
        self.test_dataset=None

        #Keep training data size
        self._train_size=None
        self._valid_size=None
        self._test_size=None

    #function to get a default transform if not specified
    def _get_default_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])
        ])

    def _get_dataset_from_name(
            self,
            train:bool=True,
            download:bool=True
    ):
        #registry for predefined datasets available through torchvision. Can also have specific args in {}
        dataset_registry={
            "cifar10":(torchvision.datasets.CIFAR10,{}),
            "cifar100":(torchvision.datasets.CIFAR100,{}),
            "mnist":(torchvision.datasets.MNIST,{}),
            "fashion_mnist":(torchvision.datasets.FashionMNIST,{}),
            "user-defined":(None,{})
        }

        if self.dataset_name not in dataset_registry:
            raise ValueError(f"Dataset '{self.dataset_name} not supported."
                             f"Choose from {list(dataset_registry)}")
        
        dataset_class, extra_args = dataset_registry[self.dataset_name]

        if self.dataset_name=="user-defined":
            #manual stuff. Come back to this later
            return None, None
        else:
            dataset_args={
                'root':str(self.data_root),
                'download':True,
                **extra_args, #dataset specific arguments that are defined in the code
                **self.kwargs #extra arguments provided by user
            }
        
        return dataset_class, dataset_args #returns dataset object and arg object
    
    def Initialize(self,
                   transform_train=None,
                   transform_test=None,
                   transform_valid=None
                   ):
        self.transforms_train=transform_train or self.transforms_train
        self.transforms_test=transform_train or self.transforms_test
        self.transforms_valid=transform_train or self.transforms_valid

        train_data_class,train_args=self._get_dataset_from_name(train=True)
        test_data_class,test_args=self._get_dataset_from_name(train=False)

        full_train_dataset=train_data_class(
            transform=self.transforms_train,
            **train_args
        )

        #create validation
        total_size=len(full_train_dataset)
        self._train_size=int(total_size*self._train_split)
        self._valid_size=total_size-self._train_size
        
        self.train_dataset,self.valid_dataset=torch.utils.data.random_split(
            full_train_dataset,
            [self._train_size,self._valid_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.test_dataset=test_data_class(
            transform=self.transforms_train,
            **test_args
        )
        self._test_size=len(self.test_dataset)

        print(f"\nData analysed:\nTrain: {self._train_size}\nValidaton: {self._valid_size}\nTest: {self._test_size}\n")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call initialize before creating dataloaders")
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=self.shuffle,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory,
                                           )
    
    def valid_dataloader(self) -> torch.utils.data.DataLoader:
        if self.valid_dataset is None:
            raise RuntimeError("Call initialize before creating dataloaders")
        return torch.utils.data.DataLoader(self.valid_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory,
                                           )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        if self.valid_dataset is None:
            raise RuntimeError("Call initialize before creating dataloaders")
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory,
                                           )
    
    def get_dataset_size(self)->Dict[str,int]:
        return{
            'train':self._train_size,
            'valid':self._valid_size,
            'test':self._test_size
        }
    
    def export_config(self) -> Dict[str,Any]:
        return {
            'dataset_name':self.dataset_name,
            'root':self.data_root,
            'batch_size':self.batch_size,
            'shuffle':self.shuffle,
            'num_workers':self.num_workers,
            'pin_memory':self.pin_memory,
            'valid_split':self._valid_split,
            'is_predefined_data':self.is_predefined_data,
            'dataset_sizes':self.get_dataset_size()
        }
    
    def __repr__(self) -> str: # defines how to display this class as a string for debugging/loggin
        return (f"Dataset(dataset={self.dataset_name}),"
                f"batch_size={self.batch_size},"
                )
