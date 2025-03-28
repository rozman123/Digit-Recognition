import torch

def CNN_model(device):
        a,b,c=32,64,128
        
        def conv_out_size(model):
                dummy_input = torch.randn(1, 1, 28, 28)
                output = model(dummy_input)  
                size=output.shape[1]
                return size
        
        print(f'Creating CNN model')
        layers0=[ 
                torch.nn.Conv2d(1,a,kernel_size=3,stride=2,padding=1),# 1 for grayscale
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(a),
                torch.nn.MaxPool2d(kernel_size=3, stride=2),
                torch.nn.Conv2d(a, b, kernel_size=3,stride=2,padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(b),
                torch.nn.MaxPool2d(kernel_size=3, stride=2),
                torch.nn.Flatten()
                ]
        model=torch.nn.Sequential(*layers0).to(device)
        conv_out_size=conv_out_size(model)
        layers1=[                
                torch.nn.Linear(conv_out_size,c),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(c,10)
                ]       
        layers_All=layers0+layers1
        model=torch.nn.Sequential(*layers_All).to(device)
        model=torch.compile(model)
        return model

