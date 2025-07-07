def safe_load_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if own_state[name].shape == param.shape:
                own_state[name].copy_(param)
            else:
                print(f'Skipping {name} due to size mismatch: model {own_state[name].shape}, checkpoint {param.shape}.')
        else:
            print(f'Skipping {name} as it is not in the model.')