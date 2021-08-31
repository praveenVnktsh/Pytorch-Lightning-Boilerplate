from model import Model

from dataloader import CustomDataset, LitCustomData
import pytorch_lightning as pl

if __name__ == '__main__':
    hparams = {
        'lr': 0.0019054607179632484
    }
    model = Model(hparams)

    dataset = LitCustomData("datasets/")

    trainer = pl.Trainer(gpus=1, max_epochs=100)
    trainer.fit(model, dataset)
