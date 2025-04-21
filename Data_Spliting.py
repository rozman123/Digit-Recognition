
if __name__=="__main__":  
    import pandas as pd
    import os

    listOfTuplesOfXandY=[]

    pathToFiles=os.path.dirname(os.path.abspath(__file__))+"\\Data\\All_combined\\"

    fileX_CSV=pd.read_csv(pathToFiles+"All_Data_X.csv")
    fileY_CSV=pd.read_csv(pathToFiles+"All_Lables_Y.csv")


    All_combined=pd.concat([fileX_CSV,fileY_CSV],axis=1)
    

    #print(All_combined.head())

    