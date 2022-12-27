import os
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet
from collections import defaultdict
from game_board import Board, Game
import random

base_path = 'model_15_15_human'
board_width = 15
board_height = 15
n_games = 10

def match(net1, net2):
    net1 = PolicyValueNet(board_width, board_height, 19, init_model=net1, cuda=True)
    player1 = MCTSPlayer(policy_value_function=net1.policy_value_fn_random,
                                       action_fc=net1.action_fc_test,
                                       evaluation_fc=net1.evaluation_fc2_test,
                                       c_puct=5,
                                       n_playout=40,
                                       is_selfplay=False)
    net2 = PolicyValueNet(board_width, board_height, 19, init_model=net2, cuda=True)
    player2 = MCTSPlayer(policy_value_function=net2.policy_value_fn_random,
                                       action_fc=net2.action_fc_test,
                                       evaluation_fc=net2.evaluation_fc2_test,
                                       c_puct=5,
                                       n_playout=40,
                                       is_selfplay=False)

    board = Board(width=board_width, height=board_height, n_in_row=5)
    game = Game(board)

    win_cnt = defaultdict(int)
    for i in range(n_games):
        winner = game.start_play(player1=player1, player2=player2, start_player=i % 2, is_shown=0, print_prob=False)
        win_cnt[winner] += 1
        print('Game %d: player %d win'%(i, winner))
    win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
    print("win: {}, lose: {}, tie:{}, win ratio:{:.2f}".format(win_cnt[1], win_cnt[2], win_cnt[-1], win_ratio*100))
    return win_ratio

if __name__=='__main__':
    match('model_15_15_human/epoch19_clip.pt', 'tmp/best_policy.model')
    # cands = ['epoch%d_clip.pt'%i for i in range(19, 30)]
    # while len(cands) > 1:
    #     random.shuffle(cands)
    #     print()
    #     print('+'*60, 'New Turn')
    #     print('candidates', cands)
    #     print()
    #     new_cands = []
    #     if (len(cands) % 2 == 1):
    #         new_cands.append(cands[0])
    #         cands.remove(cands[0])
    #     assert (len(cands) % 2 == 0)
    #     for i in range(0, len(cands), 2):
    #         print('-'*50, cands[i], 'v.s.', cands[i+1])
    #         win_ratio = match(os.path.join(base_path, cands[i]), os.path.join(base_path, cands[i+1]))
    #         if win_ratio > 0.5:
    #             new_cands.append(cands[i])
    #         else:
    #             new_cands.append(cands[i+1])
    #     cands = new_cands

    # print()
    # print('+'*60, 'Winner:', cands[0])
    # print()
        
    # os.system('cp %s %s'%(os.path.join(base_path, cands[0]), os.path.join(base_path, 'best_policy.model')))
