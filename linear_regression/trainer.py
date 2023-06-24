import torch

from d2l import torch as d2l

class Trainer(d2l.HyperParameters):
    def __init__(self, max_epochs, gradient_clip_val=0):
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_batch(self, batch):
        return batch
    
    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optimizer = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optimizer.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optimizer.step()
            self.train_batch_idx += 1
            if self.val_dataloader is None:
                return
            self.model.eval()
            for batch in self.val_dataloader:
                with torch.no_grad():
                    self.model.validation_step(self.prepare_batch(batch))
                self.val_batch_idx += 1

