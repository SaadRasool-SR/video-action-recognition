# Using GPU: NVIDIA GeForce RTX 4070 Laptop GPU
## Running baseline evaluation (pretrained backbone + random UCF101 head)
### Dataset Configuration: CROP_SIZE = 112, frames_per_clip = 8 
* **lr=1e-3, batch_size=8, data_set_fraction=100%, freeze=True** 
    * **Results:** Baseline accuracy on validation set: 2.084%

### Dataset Configuration: CROP_SIZE = 224, frames_per_clip = 16
* **lr=1e-3, batch_size=8, data_set_fraction=100%, freeze=True**
    * **Results:** Baseline accuracy on validation set: 1.231%

### Dataset Configuration: CROP_SIZE = 112, frames_per_clip = 8
* **lr=1e-3, batch_size=8, data_set_fraction=100%, freeze=True**
    * **Results:** Baseline accuracy on validation set: 0.941%

## Complete Dataset
### CROP_SIZE = 112, frames_per_clip = 8
* **lr=1e-4, epochs=1, batch_size=8, data_set_fraction=100%, freeze=True**
        * **note:** by passed the fraction subclass wrapper
    * Starting Training:
        * Epoch 01, train_loss=1.0211, val_acc=83.911% epoch_time=4475.12 sec
    * **Results:** Best Accuracy 83.911% 

* **lr=1e-3, epochs=1, batch_size=8, data_set_fraction=100%, freeze=True**
        * **note:** by passed the fraction subclass wrapper
    * Starting Training:
        * Epoch 01, train_loss=0.4206, val_acc=82.815% epoch_time=4759.38 sec
    * **Results:** Best Accuracy 82.815%
 
* **lr=1e-3, epochs=5, batch_size=8, data_set_fraction=100%, freeze=True**
        * **note:** by passed the fraction subclass wrapper
    * Starting Training:
        * Epoch 01, train_loss=0.4206, val_acc=82% epoch_time=3000 sec
        * Epoch 02, train_loss=0.25, val_acc=82% epoch_time=3000 sec
        * Epoch 03, train_loss=0.21, val_acc=81% epoch_time=3000 sec
        * Epoch 04, train_loss=0.20, val_acc=82.5% epoch_time=3000 sec
        * Epoch 05, train_loss=0.19, val_acc=83.0% epoch_time=3000 sec
    * **Results:** Best Training Accuracy 94% , Best Validation Accuracy 83%
      
 * **lr=1e-3, epochs=1, batch_size=8, data_set_fraction=100%, freeze=True**
        * **note:** by passed the fraction subclass wrapper
    * Starting Training:
        * Epoch 01, train_loss=0.4201, train_acc = 88.4%,  val_acc=82.28% epoch_time=3600 sec
    * **Results:** Best Validation Accuracy 82.28%
      

## Dataset Fraction at 50%
### CROP_SIZE = 224, frames_per_clip = 16
* **lr=1e-3, epochs=1, batch_size=8, data_set_fraction=50%, freeze=True**
    * Starting Training:
        * Epoch 01, train_loss=1.0605, val_acc=32.761% epoch_time=1120.40 sec    
    * **Results:** Best Accuracy 32.761%

* **lr=1e-4, epochs=1, batch_size=8, data_set_fraction=50%, freeze=True**
    * Starting Training:
        * Epoch 01, train_loss=2.2658, val_acc=22.544% epoch_time=1182.13 sec
    * **Results:** Best Accuracy 22.544%
      
* **lr=1e-3, epochs=1, batch_size=8, data_set_fraction=50%, freeze=True**
    * Starting Training:
        * Epoch 01, train_loss=0.0536, train_acc = 98%, val_acc=39.5% epoch_time=2000 sec
    * **Results:** Best Accuracy 39.5%%


## Dataset Fraction at 100%
### CROP_SIZE = 224, frames_per_clip = 16
* **lr=1e-3, epochs=5, batch_size=8, data_set_fraction=100%, freeze=True**
    * **note:** by passed the fraction subclass wrapper
    * Starting Training:
        * Epoch 01, train_loss=1.1604, val_acc=71.459% epoch_time=12125.20 sec
        * Epoch 02, train_loss=0.4897, val_acc=72.419% epoch_time=12196.21 sec
        * Epoch 03, train_loss=0.3980, val_acc=73.664% epoch_time=11914.16 sec
        * Epoch 04, train_loss=0.3536, val_acc=74.068% epoch_time=11843.65 sec
        * Epoch 05, train_loss=0.3238, val_acc=72.675% epoch_time=11850.64 sec

    * **Results:** Best Accuracy 74.068%
