import torch
import os
import scipy.io as sio
import numpy as np
from model import MLP, RISphase
from local_train import local_train
from CSI_process import split_data, normalization, sampling

def fedavg(state_dicts):
    avg_dict = {}
    for key in state_dicts[0].keys():
        avg_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
    return avg_dict


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # path = os.getcwd()
    # print(path)
    file_path = os.path.join(current_dir, "RIS_Channels_MIMO.mat")
    data = sio.loadmat(file_path)
    data = data['H'] # np.ndarray
    # print(data.shape)

    train_ratio = 0.8
    data_train, data_test = split_data(data, train_ratio)
    # plot_IQ_Distribution(data_train)

    data_train, data_test = normalization(data_train, data_test)
    # plot_IQ_Distribution(data_train)
    # plot_IQ_Distribution(data_test)

    H_real = data_train.real
    H_imag = data_train.imag
    H = np.stack([H_real, H_imag], axis=2) # 合并实部虚部
    # print(H.shape)
    
    H = np.transpose(H, (3, 0, 1, 2)) # 把实例数移到一维
    print(H.shape) # (8000, 64, 4, 2)

    client_num = 10 # 10 个UE
    client_data = []
    client_data = sampling(H, client_num=10)
    # print(client_data)
    # print(type(client_data))
    # print(client_data[0])

    # parameters
    global_model = MLP()
    num_rounds = 50
    local_epochs = 10
    

    # server(bS)
    avg_local_losses_sign = []
    for rnd in range(num_rounds):
        local_states = []
        local_losses = []

        for k in range(client_num):
            local_model = MLP()
            local_model.load_state_dict(global_model.state_dict())

            state, loss = local_train(
                local_model,
                client_data[k],
                epochs=local_epochs,
                signSGD=True
            )

            local_states.append(state)
            local_losses.append(loss)

        # 聚合
        global_state = fedavg(local_states)
        global_model.load_state_dict(global_state)

        print(f"Round {rnd}: Avg local loss = {np.mean(local_losses):.4f}")

        avg_local_losses_sign.append(np.mean(local_losses))
