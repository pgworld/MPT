import torch, os

def load_model_statedict(model_structure, model_path):
    if not os.path.isfile(model_path):
        raise ValueError(f'Invalid model path: {model_path}')

    ckpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = {}

    tmp_state_dict = ckpoint
    for i in tmp_state_dict:
        if i.startswith('modult') and not i.startwith('module_list'):
            state_dict[i[7:]] = tmp_state_dict[i]
        else:
            state_dict[i] = tmp_state_dict[i]

    model_structure.load_state_dict(state_dict, strict=False)

    return model_structure

def element_prod(lst):
    ret = 1
    for el in lst:
        ret *= el
    return ret
