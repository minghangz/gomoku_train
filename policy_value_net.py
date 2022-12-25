import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


class PolicyValueNet(nn.Module):
    def __init__(self, board_width, board_height, block, init_model=None, transfer_model=None, cuda=False):
        super().__init__()
        print()
        print('building network ...')
        print()

        self.planes_num = 9 # feature planes
        self.nb_block = block # resnet blocks
        if cuda == False:
            # use GPU or not ,if there are a few GPUs,it's better to assign GPU ID
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.device = 'cuda' if cuda else 'cpu'

        self.board_width = board_width
        self.board_height = board_height

        # Network
        layers = [nn.Conv2d(self.planes_num, 64, 1),
                  nn.BatchNorm2d(64),
                  nn.ReLU(inplace=True)]
        for i in range(self.nb_block):
            layers.append(BasicBlock(64))
        self.body = nn.Sequential(*layers)

        self.action_net = nn.Sequential(
            nn.Conv2d(64, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2*self.board_height*self.board_width, self.board_height*self.board_width),
            nn.LogSoftmax(dim=-1)
        )
        self.value_net = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.board_height*self.board_width, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()
        )

        if init_model is not None:
            net_params = torch.load(init_model, map_location='cpu')
            self.load_state_dict(net_params)
            print('model loaded!')
        elif transfer_model is not None:
            net_params = torch.load(transfer_model, map_location='cpu')
            self.load_state_dict({key:value for key, value in net_params.items() if 'body' in key }, strict=False)
            print('transfer model loaded !')
        else:
            print('can not find saved model, learn from scratch !')

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

        self.action_fc_test = None
        self.evaluation_fc2_test = None
        self.to(self.device)
    
    def forward(self, input_states):
        feats = self.body(input_states)
        action = self.action_net(feats)
        value = self.value_net(feats)

        return action, value
    
    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        self.optimizer.zero_grad()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        mcts_probs = torch.FloatTensor(np.array(mcts_probs)).to(self.device)
        winner_batch = torch.FloatTensor(np.array(winner_batch)).to(self.device).view(-1, 1)
        action, value = self(state_batch)
        value_loss = F.mse_loss(value, winner_batch)
        policy_loss = -(mcts_probs * action).sum(dim=-1).mean()
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()

        entropy = -(action.exp() * action).sum(dim=-1).mean()
        
        return loss.detach(), entropy.detach()
    
    def policy_value(self, state_batch, *args, **kwargs):
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        log_act_probs, value = self(state_batch)
        act_probs = np.exp(log_act_probs.detach().cpu().numpy())
        return act_probs, value.detach().cpu().numpy()

    def policy_value_fn(self, board, *args, **kwargs):
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, self.planes_num, self.board_width, self.board_height))
        act_probs, value = self.policy_value(current_state)
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value

    def policy_value_fn_random(self, board, *args, **kwargs):
        '''
        input: board,actin_fc,evaluation_fc
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        '''
        # like paper said,
        # The leaf node sL is added to a queue for neural network
        # evaluation, (di(p), v) = fθ(di(sL)),
        # where di is a dihedral reflection or rotation
        # selected uniformly at random from i in [1..8]

        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, self.planes_num, self.board_width, self.board_height))

        # print('current state shape',current_state.shape)

        #add dihedral reflection or rotation
        rotate_angle = np.random.randint(1, 5)
        flip = np.random.randint(0, 2)
        equi_state = np.array([np.rot90(s, rotate_angle) for s in current_state[0]])
        if flip:
            equi_state = np.array([np.fliplr(s) for s in equi_state])
        # print(equi_state.shape)

        # put equi_state to network
        act_probs, value = self.policy_value(np.array([equi_state]))

        # get dihedral reflection or rotation back
        equi_mcts_prob = np.flipud(act_probs[0].reshape(self.board_height, self.board_width))
        if flip:
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
        equi_mcts_prob = np.rot90(equi_mcts_prob, 4 - rotate_angle)
        act_probs = np.flipud(equi_mcts_prob).flatten()

        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value
    
    def get_policy_param(self):
        net_params = self.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)

    def restore_model(self, model_file):
        net_params = torch.load(model_file, map_location=self.device)
        self.load_state_dict(net_params)

    @property
    def network_all_params(self):
        return self.state_dict()
    
    def save_numpy(self, params):
        print('saving model as numpy form ...')
        torch.save(params, 'tmp/model.npy')
    
    def load_numpy(self, params, path='tmp/model.npy'):
        print('loading model from numpy form ...')
        params = torch.load(path, map_location=self.device)
        self.load_state_dict(params)
        print('load model from numpy!')


class BasicBlock(nn.Module):
    def __init__(self, planes):
        super().__init__()

        self.conv1 = nn.Conv2d(planes, planes, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out