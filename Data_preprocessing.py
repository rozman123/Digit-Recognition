import idx2numpy
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt

if __name__=='__main__':
    def batch_load(path_to_digit_folder,batch,transform,folder,jpg_lables,jpg_data):
        
        for file in batch:
            path_to_jpg_image=path_to_digit_folder+'/'+file
            jpg_lables.append(folder)
            image_loaded=Image.open(path_to_jpg_image)
            trans_image=transform(image_loaded).view(-1).numpy()
            jpg_data.append(trans_image)

        return (jpg_lables,jpg_data)

    def files_jpg_load(path_to_jpg_files,batch_size,transform):

        jpg_lables=[]
        jpg_data=[]

        for folder in os.listdir(path_to_jpg_files):
            print("I am processing digit: ",folder)
            path_to_digit_folder=path_to_jpg_files+folder
            all_files=[ file for file in os.listdir(path_to_digit_folder) if file.endswith('.jpg')]
            number_of_jpg_files=len(all_files)
            number_of_batches=(number_of_jpg_files//batch_size)+1

            start=0
            for nr_batch in range(number_of_batches):
                print("I am processing batch nr: ",nr_batch+1)
                end=min((nr_batch+1)*batch_size,number_of_jpg_files)
                batch=all_files[start:end]
                (jpg_lables,jpg_data)=batch_load(path_to_digit_folder,batch,transform,folder,jpg_lables,jpg_data)
                start=end

        return (jpg_lables,jpg_data)

##################################################################################
##################################################################################

data_path = os.path.dirname(os.path.abspath(__file__))+'/Data/'

if __name__=='__main__':

    print(f'Reading csv file')
    csv_file=pd.read_csv(data_path+'csv/MNIST_Digits.csv')
    csv_file.rename(columns={f'pixel{i}':i for i in range(0,784)},inplace=True)
    X_csv=pd.DataFrame(csv_file.drop(['label'],axis=1))
    Y_csv=pd.DataFrame(csv_file['label'])

    print(f'Reading idx files, joining them and turning to Dataframes')
    idx_train_image=idx2numpy.convert_from_file(data_path+'idx/train-images.idx3-ubyte')
    idx_train_lable=idx2numpy.convert_from_file(data_path+'idx/train-labels.idx1-ubyte')

    idx_test_image=idx2numpy.convert_from_file(data_path+'idx/t10k-images.idx3-ubyte')
    idx_test_lable=idx2numpy.convert_from_file(data_path+'idx/t10k-labels.idx1-ubyte')

    numpy_train_image_data_idx=np.array(idx_train_image,dtype=np.float64)
    numpy_test_image_data_idx=np.array(idx_test_image,dtype=np.float64)


    X_train_idx=pd.DataFrame(numpy_train_image_data_idx.reshape(-1,28*28))
    X_train_idx.rename(columns={i:i for i in range(0,784)},inplace=True)
    X_test_idx=pd.DataFrame(numpy_test_image_data_idx.reshape(-1,28*28))
    X_test_idx.rename(columns={i:i for i in range(0,784)},inplace=True)


    Y_train_idx=pd.DataFrame(idx_train_lable)
    Y_train_idx.rename(columns={0:'label'},inplace=True)
    Y_test_idx=pd.DataFrame(idx_test_lable)
    Y_test_idx.rename(columns={0:'label'},inplace=True)


    temp_csv_idx_Data=pd.concat([X_csv,X_train_idx,X_test_idx],ignore_index=True)
    temp_csv_idx_Lables=pd.concat([Y_csv,Y_train_idx,Y_test_idx],ignore_index=True)
    print(f'Idx and csv data join together in Dataframes')

    path_to_jpg_files = data_path+'jpg/'

    transform=transforms.Compose([
        transforms.Resize((28,28)),
        transforms.RandomRotation(degrees=10), # Rotate images
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Shifts image left and right
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)), # Rescales the image
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Applays random brightnes and contrast
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), # Makes images blur
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalizes the data
        ])

    batch_size=500

    print(f'Loading jpg files')
    (jpg_lables,jpg_data)=files_jpg_load(path_to_jpg_files,batch_size,transform)
    print(f'Jpg files loaded')

    jpg_numpy_array_data=np.array(jpg_data)
    jpg_numpy_array_lables=np.array(jpg_lables)

    jpg_Data_frame_data=pd.DataFrame(jpg_numpy_array_data)
    jpg_Data_frame_lables=pd.DataFrame(jpg_numpy_array_lables)
    jpg_Data_frame_lables.rename(columns={0:'label'},inplace=True)


    # image1=jpg_Data_frame_data.loc[21145].values.reshape(28, 28)
    # print(jpg_Data_frame_lables.loc[21145])
    # image2=jpg_Data_frame_data.loc[1145].values.reshape(28, 28)
    # print(jpg_Data_frame_lables.loc[1145])

    # plt.imshow(image1)
    # plt.show()
    # plt.imshow(image2)
    # plt.show()

    print(f'Joining jpg, idx and csv data')
    All_Data_X=pd.concat([temp_csv_idx_Data,jpg_Data_frame_data],ignore_index=True)
    All_Lables_Y=pd.concat([temp_csv_idx_Lables,jpg_Data_frame_lables],ignore_index=True)

    print(All_Data_X)
    print(All_Lables_Y)

    print(f'Saving all data to big csv')
    path_to_All_combined_data=data_path+'All_combined/All_Data_X.csv'
    path_to_All_combined_lables=data_path+'All_combined/All_Lables_Y.csv'
    All_Data_X.to_csv(path_to_All_combined_data,index=False,index_label=False)
    All_Lables_Y.to_csv(path_to_All_combined_lables,index=False,index_label=False)
    print(f'Data saved to csv')
    print(f'End of data preprocessing')
