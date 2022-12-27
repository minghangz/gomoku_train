import torch
from policy_value_net import PolicyValueNet
from create_dataset import Gomocup
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

base_path = 'gomocup'
epochs = 30

class Metric:
    def __init__(self) -> None:
        self.data = 0
        self.num = 0

    def update(self, value, n):
        self.data += value * n
        self.num += n
    
    def val(self):
        return self.data / self.num


if __name__=='__main__':
    model = PolicyValueNet(20, 20, 19, init_model='model_15_15_human/epoch19_acc62.34.pt', cuda=True)
    model.train()

    inputs = np.load(os.path.join(base_path, 'inputs.npy'))
    outputs = np.load(os.path.join(base_path, 'outputs.npy'))
    winners = np.load(os.path.join(base_path, 'winners.npy'))

    datasize = inputs.shape[0]
    train_size = int(datasize * 0.8)
    train_data = Gomocup(inputs[:train_size], outputs[:train_size], winners[:train_size], is_train=True)
    val_data = Gomocup(inputs[train_size:], outputs[train_size:], winners[train_size:], is_train=False)
    train_loader = DataLoader(train_data, batch_size=1024, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=1024, shuffle=False, num_workers=8)

    Path('model_15_15_human').mkdir(exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, threshold=1e-3)
    for epoch in tqdm(range(20, epochs)):
        loss_metric = Metric()
        entropy_metric = Metric()
        acc_metric = Metric()
        it = 0
        for data in tqdm(train_loader, desc='Train'):
            optimizer.zero_grad()

            input_data, output_data, winner_data = data
            input_data = input_data.float().cuda()
            output_data = output_data.float().cuda()
            winner_data = winner_data.float().cuda().view(-1, 1)

            action, value = model(input_data)
            value_loss = F.mse_loss(value, winner_data)
            policy_loss = -(output_data * action).sum(dim=-1).mean()
            loss = value_loss + policy_loss

            loss.backward()
            optimizer.step()

            entropy = -(action.exp() * action).sum(dim=-1).mean()
            action_acc = (action.argmax(dim=-1) == output_data.argmax(dim=-1)).float().mean()

            loss_metric.update(loss.item(), input_data.size(0))
            entropy_metric.update(entropy.item(), input_data.size(0))
            acc_metric.update(action_acc.item(), input_data.size(0))

            if (it + 1) % 100 == 0:
                print('It %d: acc: %.2f, loss: %.2f, entropy: %.2f'%(it, acc_metric.val()*100, loss_metric.val(), entropy_metric.val()))
                loss_metric = Metric()
                entropy_metric = Metric()
                acc_metric = Metric()
            it += 1
        
        loss_metric = Metric()
        entropy_metric = Metric()
        acc_metric = Metric()

        with torch.no_grad():
            for data in tqdm(val_loader, desc='Evaluate'):
                input_data, output_data, winner_data = data
                input_data = input_data.float().cuda()
                output_data = output_data.float().cuda()
                winner_data = winner_data.float().cuda().view(-1, 1)

                action, value = model(input_data)
                value_loss = F.mse_loss(value, winner_data)
                policy_loss = -(output_data * action).sum(dim=-1).mean()
                loss = value_loss + policy_loss
                entropy = -(action.exp() * action).sum(dim=-1).mean()
                action_acc = (action.argmax(dim=-1) == output_data.argmax(dim=-1)).float().mean()

                loss_metric.update(loss.item(), input_data.size(0))
                entropy_metric.update(entropy.item(), input_data.size(0))
                acc_metric.update(action_acc.item(), input_data.size(0))
        
        scheduler.step(acc_metric.val())
        
        print('Epoch %d: acc: %.2f, loss: %.2f, entropy: %.2f'%(epoch, acc_metric.val()*100, loss_metric.val(), entropy_metric.val()))
        torch.save(model.state_dict(), os.path.join('model_15_15_human', 'epoch%d_acc%.2f.pt'%(epoch, acc_metric.val()*100)))

