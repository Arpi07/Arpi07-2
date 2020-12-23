# Arpi07-2


The entire architecture consists of two stream. One is main stream and the other part is shape stream. The dual task loss function is used to train the entire model.


Follwing steps are needed to use the code- 

1. Create one root directory named Attention_cnn. Then create another three sub-directories, namely, Data, model and train within that root directory. Also, keep datasets_utils.py under root directory.

2. Keep Get_Dataset.py, Data_Process.py and Get_Dataset_utils.py within Data sub-directory.

3. Keep Resnet50_m.py, Network.py and Def_Model.py within model sub-directory.

4. Keep Data_Prep.py, training_utils.py, Loss_Dual.py, Train_and_Eval.py and Train_Model.py within train sub-directory. 


Packages required to run the following code:

1.PIL

2.Tensorflow 2.0

3. Keras

4. Matplotlib

5. Numpy

6. Scipy

7. imageio



##Training on the dataset...


Buidling network..

import Attention_cnn.model

number_classes = 3 

#Speech balloons, narrative text boxes and background

model = Attention_cnn.model.ACNN(n_classes=number_classes)

output = model(some_input)

logits, shape_head = output[..., :-1], output[..., -1:]


#Training model...

from Attention_cnn.train import train_model

train_model(

    n_classes=instance_of_n_classes,
    
    train_data= dataset_loader.build_training_dataset(),
    
    val_data=dataset_loader.build_validation_dataset(),
    
    optimiser=optimiser,
    
    epochs=2000,
    
    log_dir='logsBalloon',
    
    model_dir='logsBalloon/model',
    
    accum_iterations=4,
    
    loss_weights=(17., 1.)):
    
    
    

#Creating own data....

for img, label, edge_label in dataset:

    # img         [b, h, w, 3]       tf.float32 
    
    # label      [b, h, w, classes] tf.float32, classes = number of target classes
    
    # edge_label [b, h, w, 2]       tf.float32,  2 = boundary edge pixels/non-boundary edge pixels
    
   pass
   
   
   
   


#Conversion of model into saved model....


from Attention_cnn.model import export_model,ACNNInfer



export_model(

    classes=num_classes, 
    
    ckpt_path='/path/Inpput/weights', 
    
    out_dir='/dir/Output/save/model/',)




# can resize image ...


model = ACNNInfer('/dir/output/save/model/', resize=None)

seg, shape_head = model(path_to_image)
    
    
    
    


