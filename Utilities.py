from torchmetrics.classification import Accuracy,Precision,Recall,F1Score
from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
import torch
import os

def data_load(batch_size,device):
        data_path=os.path.dirname(os.path.abspath(__file__))+'/Data/'

        path_to_All_combined_data=data_path+'All_combined/All_Data_X.csv'
        path_to_All_combined_lables=data_path+'All_combined/All_Lables_Y.csv'

        print(f'Reading csv file with all Data')
        X_Data_file=pd.read_csv(path_to_All_combined_data)
        Y_Lable_file=pd.read_csv(path_to_All_combined_lables)

        X_train,X_eval,Y_train,Y_eval=train_test_split(X_Data_file,Y_Lable_file,test_size=0.3,train_size=0.7)

        X_train=torch.tensor(X_train.values,dtype=torch.float32,device=device)
        X_eval=torch.tensor(X_eval.values,dtype=torch.float32,device=device)

        Y_train=torch.tensor(Y_train.values,dtype=torch.long,device=device).squeeze().to(torch.long)
        Y_eval=torch.tensor(Y_eval.values,dtype=torch.long,device=device).squeeze().to(torch.long)

        X_train=X_train.view(-1,1,28,28).to(device)
        X_eval=X_eval.view(-1,1,28,28).to(device)

        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        eval_dataset = torch.utils.data.TensorDataset(X_eval, Y_eval)

        train_data=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        eval_data=torch.utils.data.DataLoader(eval_dataset,batch_size=batch_size,shuffle=False)

        return train_data,eval_data


def torch_metrics_initialize(device):

        Accuracy_fun=Accuracy(task='multiclass',average='macro',num_classes=10).to(device)
        Precision_fun=Precision(task='multiclass',average='macro',num_classes=10).to(device)
        Recall_fun=Recall(task='multiclass',average='macro',num_classes=10).to(device)
        F1_fun=F1Score(task='multiclass',average='macro',num_classes=10).to(device)

        return  Accuracy_fun,Precision_fun,Recall_fun,F1_fun


def metrics_reset(Accuracy_fun,Precision_fun,Recall_fun,F1_fun):
        Accuracy_fun.reset()
        Recall_fun.reset()
        Precision_fun.reset()
        F1_fun.reset()


def metrics_compute(Accuracy_fun,Precision_fun,Recall_fun,F1_fun):

        Accuracy_=Accuracy_fun.compute()
        Recall_=Recall_fun.compute()
        Precision_=Precision_fun.compute()
        F1_=F1_fun.compute()

        return Accuracy_,Recall_,Precision_,F1_

def metrics_update(output,target,Accuracy_fun,Precision_fun,Recall_fun,F1_fun)->None:
        Accuracy_fun.update(output,target)
        Recall_fun.update(output,target)
        Precision_fun.update(output,target)
        F1_fun.update(output,target)



def train(model,train_data,loss_function,optimalizer,device):
        model.train().to(device)
        Total_num_of_batches=len(train_data)
        num_of_batches=0
        for X,Y in train_data:
                        
                X,Y=X.to(device),Y.to(device)
                output=model(X)
                optimalizer.zero_grad()
                Loss_=loss_function(output,Y)
                Loss_.backward()
                optimalizer.step()

                num_of_batches+=1

                if not num_of_batches % 500:
                        print(f'Batch training progres: {(num_of_batches/Total_num_of_batches*100):.2f}%')
                        print(f'Batch Loss: {Loss_:.4f}')

        return model


def eval(model,eval_data,loss_function,device,epoch):    
        print(f'Validating the model')
        Validation_Loss_ = 0
        Accuracy_fun,Precision_fun,Recall_fun,F1_fun=torch_metrics_initialize(device)
        number_of_validation_elements=0
        with torch.no_grad():  # Off the gradient tracking
                model.eval().to(device)

                for X,Y in eval_data:
                        
                        number_of_validation_elements+=len(Y)
                        X,Y=X.to(device),Y.to(device)
                        pred=model(X)
                        metrics_update(pred,Y,Accuracy_fun,Precision_fun,Recall_fun,F1_fun)
                        Validation_Loss_+=loss_function(pred,Y).item()
        
        Accuracy_,Recall_,Precision_,F1_=metrics_compute(Accuracy_fun,Precision_fun,Recall_fun,F1_fun)
        
        Validation_Metrics_string=creates_string_with_validation_metrics(Accuracy_,Recall_,Precision_,F1_,Validation_Loss_,len(eval_data),number_of_validation_elements,epoch)

        return (model,Validation_Metrics_string)

def creates_string_with_validation_metrics(Accuracy_,Recall_,Precision_,F1_,Validation_Loss_,data_length,number_of_validation_elements,epoch):

        Validation_Metrics_string=f'################## Validation ##################\nValidation Metrics over {number_of_validation_elements} validation instances and after {epoch} epoch of training:\nAverage Accurecy: {Accuracy_:.4f}\nAverage Recall: {Recall_:.4f}\nAverage Precision: {Precision_:.4f}\nAverage F1: {F1_:.4f}\nAverage Loss: {(Validation_Loss_/data_length):.4f}\n' 
        ## string with metrics for saving

        return Validation_Metrics_string


def save_model(model,Validation_Metrics_string):
        print(f'Saving the model')
        now=datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        saved_model_name=f'digite_recognition_model_3_Datasets_{now}.pth'
        index_to_append_name_of_the_model=Validation_Metrics_string.find('\n')
        Validation_Metrics_string_and_model_name=Validation_Metrics_string[:index_to_append_name_of_the_model]+'\n'+saved_model_name+Validation_Metrics_string[index_to_append_name_of_the_model:]+'\n\n'
        model_folder_path=os.path.dirname(os.path.abspath(__file__))+'/model/'
        model_saving_path=model_folder_path+saved_model_name
        torch.save(model.state_dict(),model_saving_path)
        print(f'Saving the metrics')
        with open(model_folder_path+'Models Metrics.txt','a') as file:
                file.write(Validation_Metrics_string_and_model_name)
        print(f'Model saved')

