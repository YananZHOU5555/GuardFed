# 所有算法的实现
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import numpy as np

DEVICE = torch.device('cpu')
BATCH_SIZE = 256
LEARNING_RATE = 0.01
LOCAL_EPOCHS = 1

def train_fedavg(data_dict, attack_type, malicious_ids, num_rounds, apply_attack_func, prepare_client_data_func, evaluate_model_func, SimpleMLP):
    clients_data = prepare_client_data_func(data_dict, attack_type, malicious_ids)
    global_model = SimpleMLP(input_size=data_dict['data_info']['num_features']).to(DEVICE)

    for round_idx in range(num_rounds):
        client_updates = []
        for i, client_data in enumerate(clients_data):
            local_model = copy.deepcopy(global_model)
            optimizer = optim.SGD(local_model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()
            local_model.train()
            dataset = TensorDataset(client_data['X'], client_data['y'])
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            for epoch in range(LOCAL_EPOCHS):
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    loss = criterion(local_model(X_batch), y_batch)
                    loss.backward()
                    optimizer.step()
            update = {name: param.data - global_model.state_dict()[name] for name, param in local_model.named_parameters()}
            if client_data['is_malicious']:
                update = apply_attack_func(update, attack_type, i, len(malicious_ids))
            client_updates.append(update)

        aggregated_update = {name: torch.stack([u[name] for u in client_updates]).mean(dim=0) for name in global_model.state_dict()}
        new_state = global_model.state_dict()
        for name in new_state:
            new_state[name] = new_state[name] + aggregated_update[name]
        global_model.load_state_dict(new_state)

    acc, aeod, aspd = evaluate_model_func(global_model, data_dict['test_loader'], DEVICE)
    return {'accuracy': acc, 'aeod': aeod, 'aspd': aspd}

def train_median(data_dict, attack_type, malicious_ids, num_rounds, apply_attack_func, prepare_client_data_func, evaluate_model_func, SimpleMLP):
    clients_data = prepare_client_data_func(data_dict, attack_type, malicious_ids)
    global_model = SimpleMLP(input_size=data_dict['data_info']['num_features']).to(DEVICE)

    for round_idx in range(num_rounds):
        client_updates = []
        for i, client_data in enumerate(clients_data):
            local_model = copy.deepcopy(global_model)
            optimizer = optim.SGD(local_model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()
            local_model.train()
            dataset = TensorDataset(client_data['X'], client_data['y'])
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            for epoch in range(LOCAL_EPOCHS):
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    loss = criterion(local_model(X_batch), y_batch)
                    loss.backward()
                    optimizer.step()
            update = {name: param.data - global_model.state_dict()[name] for name, param in local_model.named_parameters()}
            if client_data['is_malicious']:
                update = apply_attack_func(update, attack_type, i, len(malicious_ids))
            client_updates.append(update)

        aggregated_update = {name: torch.median(torch.stack([u[name] for u in client_updates]), dim=0)[0] for name in global_model.state_dict()}
        new_state = global_model.state_dict()
        for name in new_state:
            new_state[name] = new_state[name] + aggregated_update[name]
        global_model.load_state_dict(new_state)

    acc, aeod, aspd = evaluate_model_func(global_model, data_dict['test_loader'], DEVICE)
    return {'accuracy': acc, 'aeod': aeod, 'aspd': aspd}

def train_fairfed(data_dict, attack_type, malicious_ids, num_rounds, apply_attack_func, prepare_client_data_func, evaluate_model_func, SimpleMLP, beta=1.0):
    clients_data = prepare_client_data_func(data_dict, attack_type, malicious_ids)
    global_model = SimpleMLP(input_size=data_dict['data_info']['num_features']).to(DEVICE)

    for round_idx in range(num_rounds):
        client_updates = []
        client_fairness = []

        for i, client_data in enumerate(clients_data):
            local_model = copy.deepcopy(global_model)
            optimizer = optim.SGD(local_model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()
            local_model.train()
            dataset = TensorDataset(client_data['X'], client_data['y'])
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            for epoch in range(LOCAL_EPOCHS):
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    loss = criterion(local_model(X_batch), y_batch)
                    loss.backward()
                    optimizer.step()

            # 计算公平性
            local_model.eval()
            with torch.no_grad():
                outputs = local_model(client_data['X'])
                _, preds = torch.max(outputs, 1)
                sex_np = client_data['sex'].cpu().numpy() if torch.is_tensor(client_data['sex']) else client_data['sex']
                y_np = client_data['y'].cpu().numpy()
                preds_np = preds.cpu().numpy()

                mask_0_pos = (sex_np == 0) & (y_np == 1)
                tpr_0 = (preds_np[mask_0_pos] == 1).sum() / mask_0_pos.sum() if mask_0_pos.sum() > 0 else 0
                mask_1_pos = (sex_np == 1) & (y_np == 1)
                tpr_1 = (preds_np[mask_1_pos] == 1).sum() / mask_1_pos.sum() if mask_1_pos.sum() > 0 else 0
                eod = abs(tpr_0 - tpr_1)
                client_fairness.append(eod)

            update = {name: param.data - global_model.state_dict()[name] for name, param in local_model.named_parameters()}
            if client_data['is_malicious']:
                update = apply_attack_func(update, attack_type, i, len(malicious_ids))
            client_updates.append(update)

        # FairFed聚合
        fairness_weights = [np.exp(-beta * f) for f in client_fairness]
        total_weight = sum(fairness_weights)
        fairness_weights = [w / total_weight for w in fairness_weights]

        aggregated_update = {}
        for name in global_model.state_dict():
            aggregated_update[name] = sum(w * u[name] for w, u in zip(fairness_weights, client_updates))

        new_state = global_model.state_dict()
        for name in new_state:
            new_state[name] = new_state[name] + aggregated_update[name]
        global_model.load_state_dict(new_state)

    acc, aeod, aspd = evaluate_model_func(global_model, data_dict['test_loader'], DEVICE)
    return {'accuracy': acc, 'aeod': aeod, 'aspd': aspd}

def train_fltrust(data_dict, attack_type, malicious_ids, num_rounds, apply_attack_func, prepare_client_data_func, evaluate_model_func, SimpleMLP, server_lr=0.01):
    clients_data = prepare_client_data_func(data_dict, attack_type, malicious_ids)
    global_model = SimpleMLP(input_size=data_dict['data_info']['num_features']).to(DEVICE)

    # 服务器数据
    num_server = int(len(data_dict['X_train']) * 0.1)
    server_indices = np.random.choice(len(data_dict['X_train']), num_server, replace=False)
    X_server = data_dict['X_train'][server_indices]
    y_server = data_dict['y_train'][server_indices]

    for round_idx in range(num_rounds):
        # 服务器更新
        server_model = copy.deepcopy(global_model)
        optimizer = optim.SGD(server_model.parameters(), lr=server_lr)
        criterion = nn.CrossEntropyLoss()
        server_model.train()
        dataset = TensorDataset(X_server, y_server)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(server_model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        server_update = {name: param.data - global_model.state_dict()[name] for name, param in server_model.named_parameters()}

        # 客户端更新
        client_updates = []
        for i, client_data in enumerate(clients_data):
            local_model = copy.deepcopy(global_model)
            optimizer = optim.SGD(local_model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()
            local_model.train()
            dataset = TensorDataset(client_data['X'], client_data['y'])
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            for epoch in range(LOCAL_EPOCHS):
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    loss = criterion(local_model(X_batch), y_batch)
                    loss.backward()
                    optimizer.step()
            update = {name: param.data - global_model.state_dict()[name] for name, param in local_model.named_parameters()}
            if client_data['is_malicious']:
                update = apply_attack_func(update, attack_type, i, len(malicious_ids))
            client_updates.append(update)

        # 计算信任评分
        trust_scores = []
        for update in client_updates:
            server_vec = torch.cat([server_update[name].flatten() for name in server_update])
            client_vec = torch.cat([update[name].flatten() for name in update])
            cos_sim = torch.nn.functional.cosine_similarity(server_vec.unsqueeze(0), client_vec.unsqueeze(0))
            trust_scores.append(max(0, cos_sim.item()))

        total_trust = sum(trust_scores)
        if total_trust > 0:
            trust_scores = [s / total_trust for s in trust_scores]
        else:
            trust_scores = [1.0 / len(trust_scores)] * len(trust_scores)

        aggregated_update = {}
        for name in global_model.state_dict():
            aggregated_update[name] = sum(w * u[name] for w, u in zip(trust_scores, client_updates))

        new_state = global_model.state_dict()
        for name in new_state:
            new_state[name] = new_state[name] + aggregated_update[name]
        global_model.load_state_dict(new_state)

    acc, aeod, aspd = evaluate_model_func(global_model, data_dict['test_loader'], DEVICE)
    return {'accuracy': acc, 'aeod': aeod, 'aspd': aspd}
