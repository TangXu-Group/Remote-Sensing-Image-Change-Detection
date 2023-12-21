from img_read import  Data
from  EATDernet import EATDer
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from imgetest import innetimgtest




root=  'D:/djy/CDD256'
#root=  '/home/amax/djy/LEVIR-CD256'


net1 =EATDer()
#net1=nn.DataParallel(net1) 多卡训练
net1=net1.cuda()

criterion1=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]).cuda())
criterion2=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]).cuda())
lr=0.001
epochs=60
epochslim=100
batch_size2=10
phi=0.3
train_set = Data(root,'val', data_enhancement=True)
train_loader = DataLoader(train_set, batch_size=batch_size2, shuffle=True,pin_memory=True)



def train(epochs,lr,train_loader,val):
    
    bestval=9999999
    bestpre=0
    bestrecall=0
    bestf1=0
    bestacc=0
    
    
    optimizer=optim.AdamW(net1.parameters(), lr)
    for epoch in range(epochs):
        
        net1.train()
        
        
        if epoch%10==0:#每6代下降一次学习率   加快收敛
            lr=lr/2
            optimizer=optim.AdamW(net1.parameters(), lr)
            
        
            
        for i, data in enumerate(train_loader):
            
          
               
            imgs_A, imgs_B, edges,labels= data
            optimizer.zero_grad()
            edge,block= net1(imgs_A.cuda(),imgs_B.cuda())

            loss=(1-phi)*criterion1(edge,edges.cuda())+phi*criterion2(block,labels.cuda())
            
            loss.backward()
            optimizer.step() 
           
            if i%100==0:
                
                print('Train Epoch:{} batchnum:{} Loss:{:.6f} lr{:.6f}'.
                    format(epoch,i,loss.item(),lr))#打印训练次数
          
      
        if val==False and epoch%20==0:
            
            with torch.no_grad():
                net1.eval()
                total_acc,total_recall,total_precision,f1,lossbanch=innetimgtest(root,net1,"val",criterion2,use_edge=False)
                if lossbanch<bestval:
                    bestval=lossbanch
                    #torch.save(net1, 'bestlossmodel.pt')
                print("best loss now is",bestval)
                
                if bestf1<f1:
                    bestf1=f1
                    bestpre=total_precision
                    bestrecall=total_recall
                    bestacc=total_acc
                    torch.save(net1, 'bestf1model.pt')
                print("best acc recall pre f1 now is",bestacc,bestrecall,bestpre,bestf1)
                
                
        if val==True and epoch%5==0:
            
            with torch.no_grad():
                net1.eval()
                total_acc,total_recall,total_precision,f1,lossbanch=innetimgtest(root,net1,"val",criterion2,use_edge=False)
                if lossbanch<bestval:
                    bestval=lossbanch
                    #torch.save(net1, 'bestlossmodel.pt')
                print("best loss now is",bestval)
                
                if bestf1<f1:
                    bestf1=f1
                    bestpre=total_precision
                    bestrecall=total_recall
                    bestacc=total_acc
                    torch.save(net1, 'bestf1model.pt')
                print("best acc recall pre f1 now is",bestacc,bestrecall,bestpre,bestf1) 
                
            
    torch.save(net1, 'finalmodel.pt')
                
                
train(epochs,lr,train_loader,False)
lr=0.001
train_set = Data(root,'train',edgename="edge_slim", data_enhancement=True)
train_loader = DataLoader(train_set, batch_size=batch_size2, shuffle=True,pin_memory=True)


train(epochslim,lr,train_loader,True)
