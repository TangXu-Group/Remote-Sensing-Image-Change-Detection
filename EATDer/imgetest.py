from img_read import  Data
from torch.utils.data import DataLoader
import torch
import torchmetrics
from  torchmetrics import MetricCollection, Accuracy, Precision, Recall 
from torchmetrics.classification import BinaryF1Score


def innetimgtest(root,net,mode,criterion,use_edge):
    
    if use_edge==True:
        test_set = Data(root,mode, data_enhancement=False,use_edge=True)
    else:
        
        test_set = Data(root,mode, data_enhancement=False)
    test_loader = DataLoader(test_set, batch_size=1,pin_memory=True)
    imagesize2=256*256
    test_set = Data(root,mode, data_enhancement=False)
    test_loader = DataLoader(test_set, batch_size=1,pin_memory=True)
    
    net = net

    metric_collection = MetricCollection({ 
    'acc': Accuracy(task='binary'), 
    'prec': Precision(task='binary',num_classes=2), 
    'rec': Recall(task='binary',num_classes=2) ,
    'f1':BinaryF1Score() 
    }) 
    
    
    net.eval()
    lossbanch=0
    #theacc=0
    for i, data in enumerate(test_loader):
        imgs_A, imgs_B,edge, labels= data
        edge,output=net(imgs_A.cuda(), imgs_B.cuda())
        
        
        loss=float(criterion(output,labels.cuda()))
        lossbanch=loss+lossbanch
        
        output=torch.sigmoid(output)
        output=torch.where(output>0.5,1,0).int()
        labels=torch.where(labels>0.5,1,0).int()
        output=torch.reshape(output,(1,imagesize2))
       
        
        labels=torch.reshape(labels,(1,imagesize2))
        
      
        
        batch_metrics = metric_collection.forward(output[0].cpu(), labels[0].cpu()) 
  
        
     
    val_metrics = metric_collection.compute() 
    #print(f"Metrics on all data: {val_metrics}") 
    metric_collection.reset()
   
   
    
   
    return val_metrics['acc'], val_metrics['rec'], val_metrics['prec'], val_metrics['f1'],lossbanch


