import Utilities
import My_Models
import torch
import torch.optim.lr_scheduler 


device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

epoch=100
batch_size=128
alpha=0.01

train_data,eval_data=Utilities.data_load(batch_size,device)
model=My_Models.CNN_model(device)

loss_function=torch.nn.CrossEntropyLoss()
optimalizer=torch.optim.Adam(model.parameters(),lr=alpha)
torch.optim.lr_scheduler#####################################


for e in range(epoch):

        Total_training_progres=e/epoch*100

        model=Utilities.train(model,train_data,loss_function,optimalizer,device) 
        model,Validation_Metrics_string=Utilities.eval(model,eval_data,loss_function,device,epoch)

        
        if not e % 1:
                print(f'################## Epoch: {e}/{epoch} ##################')
                print(f'Whole training: {Total_training_progres:.2f}%')
                print(Validation_Metrics_string)

Utilities.save_model(model,Validation_Metrics_string)
