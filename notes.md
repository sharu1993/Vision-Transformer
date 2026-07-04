Data
- Augment the data
- Apply normalization
- Play around with it

Model training
- Apply transformation
  - Auto augmentation (learns the best possible augmentation by running a small training on a model with rewards for high validation)
- Make sure you shuffle the data during training
- Optimizer: 
  - Type
  - learning rate
  - weight decay
  - momentum
  - set optimizer to zero_grad during training
  - learning rate decay (scheduler for this)
- Loss:
  - Apply label smoothing for loss (0.9 instead of 1.0)
  - Focal loss can be used for imbalanced data sets
  - BCE (Binary cross entropy) can be used for multi class classification


Evaluation:
- set torch no grad