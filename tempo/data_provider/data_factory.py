from torch.utils.data import DataLoader

from tempo.data_provider.data_loader import (Dataset_Custom, Dataset_ETT_hour,
                                             Dataset_ETT_minute,
                                             Dataset_Monash, Dataset_Pred,
                                             Dataset_TSF)

# TODO: read the code for the custom datasets
data_dict = {
    "custom": Dataset_Custom,
    "tsf_data": Dataset_TSF,
    "ett_h": Dataset_ETT_hour,
    "ett_m": Dataset_ETT_minute,
    "monash": Dataset_Monash,
}


def data_provider(args, flag, drop_last_test=True, train_all=False):
    # data_dict[args.data] determines which dataset class to instantiate later
    Data = data_dict[args.data]

    timeenc = 0 if args.embed != "timeF" else 1

    # Percent of samples to use
    percent = args.percent

    # Max time series length
    max_len = args.max_len

    # Initialize arguments for dataloaders
    if flag == "test":
        shuffle_flag = False
        drop_last = drop_last_test
    elif flag == "pred":
        shuffle_flag = False
        drop_last = False
        Data = Dataset_Pred
    elif flag == "val":
        shuffle_flag = True
        drop_last = drop_last_test
    else:
        shuffle_flag = True
        drop_last = True
    batch_size = args.batch_size

    # Initialize dataset
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        percent=percent,
        max_len=max_len,
        train_all=train_all,
        data_name=args.data_name,
    )
    print(f"data set type: {data_set}")

    # initialize data loader
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last,
    )
    return data_set, data_loader
