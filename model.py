import pytorch_lightning as pl
from torch import nn
import numpy as np
import torch

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.nn = nn.Sequential(
            nn.Linear(28*28, 1056),
            nn.ReLU(),
            nn.Linear(1056, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.Softmax(dim = 1)
        )
        self.lossfunc = nn.CrossEntropyLoss()

    def forward(self,x):
        y = self.nn(x)
        return y
    
    def configure_optimizers(self):
        LR = 1e-3
        optimizer = torch.optim.AdamW(self.parameters(),lr=LR)
        return optimizer

    def runBatch(self, batch, batch_idx):
        x = batch['input']
        y = batch['target']

        out = self(x)
        loss = self.lossfunc(out, y)
        return {
            'x' : x,
            'y' : y,
            'pred' : out,
            'loss' : loss
        }


    def training_step(self, batch, batch_idx):
        

        dic = self.runBatch(batch, batch_idx)
        loss = dic['loss']
        self.log('train_loss', loss)

        return {'loss': loss}

    def validation_step(self,batch,batch_idx):
        dic = self.runBatch(batch, batch_idx)
        loss = dic['loss']

        self.log('val_loss', loss)
        # might wanna log images here as well

        # self.logger.experiment.add_images('val_images', data, self.current_epoch)
        # data has to be a tensor of shape (n_images, channels, height, width). 
        # data can be torch.cat() on some dimension (height/width) with the target before passing into the logger.
        
        return {'loss': loss}



if __name__ == "__main__":
    model = Model()
    output = model(torch.randn(1, 28*28))