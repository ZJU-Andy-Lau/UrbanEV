import utils
import torch
from parse import parse_args

import train
import numpy as np
from utils import split_cv,create_loaders

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

if __name__ == "__main__":

    args = parse_args()
    # device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    utils.set_seed(seed=args.seed, flag=True)
    if args.use_npy and args.add_feat == 'None':
        args.add_feat = 'npy'
    feat, adj, extra_feat, time= utils.read_data(args)
    print(
        f"Running {args.model} with feat={args.feat}, pre_l={args.pred_len}, fold={args.fold}, add_feat={args.add_feat}, pred_type(node)={args.pred_type}")

    # Initialize and train model
    net = utils.load_net(args, np.array(adj), device, feat)

    train_feat, valid_feat, test_feat, train_extra_feat, valid_extra_feat, test_extra_feat,scaler = split_cv(args,
                                                                                                   time,
                                                                                                   feat,
                                                                                                   TRAIN_RATIO,
                                                                                                 VAL_RATIO,
                                                                                                   TEST_RATIO,
                                                                                                   extra_feat)
    train_loader, valid_loader, test_loader = create_loaders(train_feat, valid_feat, test_feat,
                                                             train_extra_feat, valid_extra_feat,
                                                             test_extra_feat,
                                                             args, device)
    if args.model == 'lo' or args.model == 'ar' or args.model == 'arima':
        optim = None
        loss_func =None
        args.is_train = False
        args.stat_model = True
        train_valid_feat = np.vstack((train_feat, valid_feat,test_feat[:args.seq_len+args.pred_len,:]))
        test_loader = [train_valid_feat,test_feat[args.pred_len+args.seq_len:,:]]
    else:
        optim = torch.optim.Adam(net.parameters(), weight_decay=0.00001)
        args.stat_model = False
        loss_func = torch.nn.MSELoss()
        if args.is_train:
            train.training(args, net, optim, loss_func, train_loader, valid_loader, args.fold)

    train.test(args, test_loader, feat,net,scaler)
