import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def act(x):
    return 0 if x < 0 else 1
def go(house,rock,attr):
    X=torch.tensor([house,rock,attr],dtype=torch.float32,device=device)
    Wh=torch.tensor([[0.3,0.3,0],[0.4,-0.5,1]],device=device)
    Wout=torch.tensor([-1.0,1.0],device=device)

    Zh=torch.mv(Wh,X)
    print(f'Значения сумм на нейронах скрытого поля {Zh}')

    Uh=torch.tensor([act(x) for x in Zh],dtype=torch.float32,device=device)
    print(f'Значения на выходах нейронов скрытого поля {Uh}')

    Zout=torch.dot(Wout,Uh)
    Y=act(Zout)
    print(f"Выходное значение HC:{Y}")

    return Y

house=1
rock=1
attr=0

res=go(house,rock,attr)

if res==1:
    print(True)
else:
    print(False)