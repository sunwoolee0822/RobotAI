import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import (AUROC, Accuracy, AveragePrecision, F1Score,
                          Precision, Recall)


class CoFormerModule(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg   = cfg
        self.loss_fn = nn.CrossEntropyLoss()
        self.test_outputs = []   # 예측값 저장용

        for split in ['train', 'val', 'test']:
            setattr(self, f'{split}_f1',   F1Score(task='binary'))
            setattr(self, f'{split}_auroc',AUROC(task='binary'))
            setattr(self, f'{split}_auprc',AveragePrecision(task='binary'))
            setattr(self, f'{split}_acc',  Accuracy(task='binary'))
            setattr(self, f'{split}_prec', Precision(task='binary'))
            setattr(self, f'{split}_rec',  Recall(task='binary'))

    def forward(self, batch):
        return self.model(
            batch['data'],
            batch['time'],
            batch['mask'],
            batch['static']
        )

    def _step(self, batch, split):
        logits = self(batch)
        gt     = batch['gt'].squeeze(-1).long()
        loss   = self.loss_fn(logits, gt)
        probs  = torch.softmax(logits, dim=-1)[:, 1]

        self.log(f'{split}_loss', loss,
                 on_step=True, on_epoch=True, prog_bar=True)

        for metric_name in ['f1', 'auroc', 'auprc', 'acc', 'prec', 'rec']:
            m = getattr(self, f'{split}_{metric_name}')
            m(probs, gt)
            self.log(f'{split}_{metric_name}', m,
                     on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        gt     = batch['gt'].squeeze(-1).long()
        loss   = self.loss_fn(logits, gt)
        probs  = torch.softmax(logits, dim=-1)[:, 1]
        preds  = logits.argmax(dim=-1)

        # metric 로깅
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for metric_name in ['f1', 'auroc', 'auprc', 'acc', 'prec', 'rec']:
            m = getattr(self, f'test_{metric_name}')
            m(probs, gt)
            self.log(f'test_{metric_name}', m, on_step=False, on_epoch=True)

        # 예측값 저장 (순서 = split[2] test 인덱스 순서)
        self.test_outputs.append({
            'pred': preds.detach().cpu(),
            'prob': probs.detach().cpu(),
            'gt':   gt.detach().cpu(),
        })
        return loss

    def on_validation_epoch_end(self):
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

    def on_test_epoch_end(self):
        preds = torch.cat([o['pred'] for o in self.test_outputs]).numpy()
        probs = torch.cat([o['prob'] for o in self.test_outputs]).numpy()
        gts   = torch.cat([o['gt']   for o in self.test_outputs]).numpy()
        out = self.cfg.get('pred_out', 'test_predictions.npz')
        np.savez(out, pred=preds, prob=probs, gt=gts)
        print(f"[Saved] {out}  (N={len(preds)})")
        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg['lr'],
            weight_decay=self.cfg.get('weight_decay', 1e-4)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.cfg.get('lr_factor', 0.5),
            patience=self.cfg.get('patience', 5),
            min_lr=1e-8,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        }