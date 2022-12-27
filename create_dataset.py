import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from game_board import Board
from tqdm import tqdm

class Gomocup(Dataset):
    def __init__(self, inputs, outputs, winners, is_train=True) -> None:
        super().__init__()
        # self.inputs = np.load(os.path.join(base_path, 'inputs.npy'))
        # self.outputs = np.load(os.path.join(base_path, 'outputs.npy'))
        # self.winners = np.load(os.path.join(base_path, 'winners.npy'))
        self.inputs = inputs
        self.outputs = outputs
        self.winners = winners
        self.is_train = is_train
        
        print('Dataset size:', len(self.inputs))
    
    def aug_data(self, input, output):
        output = np.flipud(output)

        k = np.random.randint(0, 4)
        input = np.rot90(input, k=k, axes=(1, 2))
        output = np.rot90(output, k=k, axes=(0, 1))

        rot = np.random.randint(0, 2)
        if rot == 1:
            input = np.flip(input, axis=1)
            output = np.flip(output, axis=0)
            
        rot = np.random.randint(0, 2)
        if rot == 1:
            input = np.flip(input, axis=2)
            output = np.flip(output, axis=1)
        
        return input, np.flipud(output)
    
    def crop(self, input, output, tgt_size=15):
        output = np.flipud(output)

        s = input.shape[-1]
        assert s > tgt_size
        while True:
            s_x = np.random.randint(0, s-tgt_size+1)
            s_y = np.random.randint(0, s-tgt_size+1)
            if output[s_x:s_x+tgt_size, s_y:s_y+tgt_size].sum() > 0:
                break
        input = input[:, s_x:s_x+tgt_size, s_y:s_y+tgt_size]
        output = output[s_x:s_x+tgt_size, s_y:s_y+tgt_size]

        return input, np.flipud(output)

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        input = self.inputs[index]
        output = self.outputs[index]
        winner = self.winners[index]
        
        if self.is_train:
            input, output = self.aug_data(input, output)
        # input, output = self.crop(input, output)

        return np.ascontiguousarray(input), np.ascontiguousarray(output.reshape(-1)), winner
    

def process_data(base_path='gomocup', rule='freestyle'):
    # roots = os.listdir(base_path)
    # roots = ['gomocup2022results', 'gomocup2021results', 'gomocup2020results']
    roots = ['gomocup2022results']
    inputs = []
    outputs = []
    winners = []
    for root in tqdm(roots):
        if 'gomocup' in root:
            for match in os.listdir(os.path.join(base_path, root)):
                if rule in match.lower() and os.path.isdir(os.path.join(base_path, root, match)):
                    for game in os.listdir(os.path.join(base_path, root, match)):
                        if game.endswith('psq'):
                            try:
                                inp, outp, w = read_psq(os.path.join(base_path, root, match, game))
                                inputs.extend(inp)
                                outputs.extend(outp)
                                winners.extend(w)
                            except:
                                print(os.path.join(base_path, root, match, game))
    inputs = np.stack(inputs, axis=0).astype(np.float16)
    outputs = np.stack(outputs, axis=0).astype(np.float16)
    winners = np.array(winners).astype(np.float16)

    np.save(os.path.join(base_path, 'inputs.npy'), inputs)
    np.save(os.path.join(base_path, 'outputs.npy'), outputs)
    np.save(os.path.join(base_path, 'winners.npy'), winners)

    print('Dataset size:', inputs.shape[0])


def read_psq(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    w, h = lines[0].split(' ')[1].strip(',').split('x')
    w, h = int(w), int(h)
    board = Board(width=w, height=h)
    board.init_board()

    lines = lines[1:]

    inputs, outputs, winners = [], [], []

    for i, line in enumerate(lines):
        if ',' not in line:
            break

        x, y, t = np.array(line.split(','), np.int8)

        if i % 2 == 0:
            player = 1
        else:
            player = 2

        input = board.current_state()
        output = np.zeros([h, w], dtype=np.int8)
        output[y-1, x-1] = 1
        winner = player

        inputs.append(input)
        outputs.append(output)
        winners.append(winner)

        # update board
        board.do_move(board.location_to_move((y-1, x-1)))
        end, winner = board.game_end()
        if end:
            break

    if end:
        winners = [1 if w == winner else -1 for w in winners]
    else:
        winners = [0] * len(winners)
    
    return inputs, outputs, winners

if __name__ == '__main__':
    process_data()
    # dataset = Gomocup()
    # train_loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)
    # import pdb
    # pdb.set_trace()
    # data = next(iter(train_loader))
    # print()