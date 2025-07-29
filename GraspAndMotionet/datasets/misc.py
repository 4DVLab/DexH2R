from typing import Dict, List
import torch


def collate_fn_general(batch: List) -> Dict:
    """ General collate function used for dataloader.
    """
    batch_data = {key: [d[key] for d in batch] for key in batch[0]}
    
    for key in batch_data:
        if torch.is_tensor(batch_data[key][0]):
            # 
            # if key in ["pos","normal","feat"]:# because random sample
            #     max_pcd_num = max([pcd_data.shape[0] for pcd_data in batch_data[key]])
            #     pcd_data_list = []
            #     for pcd_data in batch_data[key]:
            #         pcd_tensor = torch.zeros((max_pcd_num,3))
            #         pcd_tensor[:pcd_data.shape[0]] = pcd_data
            #         pcd_data_list.append(pcd_tensor)
            #     batch_data[key] = pcd_data_list
            batch_data[key] = torch.stack(batch_data[key])
        
    return batch_data


def collate_fn_squeeze_pcd_batch_grasp(batch: List) -> Dict:
    """ General collate function used for dataloader.
    This collate function is used for point-transformer
    """
    batch_data = {key: [d[key] for d in batch] for key in batch[0]}

    for key in batch_data:
        if torch.is_tensor(batch_data[key][0]):
            batch_data[key] = torch.stack(batch_data[key])

    ## squeeze the first dimension of pos and feat
    offset, count = [], 0
    for item in batch_data['pos']:
        count += item.shape[0]
        offset.append(count)
    offset = torch.IntTensor(offset)
    batch_data['offset'] = offset

    channel_dim = batch_data['pos'].shape[2]
    batch_data['pos'] = batch_data['pos'].view((-1, channel_dim))


    return batch_data
