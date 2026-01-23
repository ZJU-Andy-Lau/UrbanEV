import os
from tqdm import tqdm
import torch
import numpy as np
import utils
import pandas as pd

def training(args, net, optim, loss_func, train_loader, valid_loader, fold):
        valid_loss = 1000
        net.train()
        for _ in tqdm(range(args.epoch), desc='Training'):
            for j, data in enumerate(train_loader):
                '''
                occupancy = (batch, seq, node)
                add_tensor = (batch, seq, node)
                label = (batch, node)
                '''
                net.train()
                extra_feat = 'None'
                if args.add_feat != 'None':
                    occupancy, label, extra_feat = data
                else:
                    occupancy, label = data

                optim.zero_grad()
                predict = net(occupancy, extra_feat)
                if predict.shape != label.shape:
                    loss = loss_func(predict.unsqueeze(-1), label)
                else:
                    loss = loss_func(predict, label)
                loss.backward()
                optim.step()

            # validation
            net.eval()
            for j, data in enumerate(valid_loader):
                '''
                occupancy = (batch, seq, node)
             = (batch, seq, node)
                label = (batch, node)
                '''
                net.train()
                extra_feat = 'None'
                if args.add_feat != 'None':
                    occupancy, label, extra_feat = data
                else:
                    occupancy, label = data

                predict = net(occupancy,extra_feat)
                if predict.shape != label.shape:
                    loss = loss_func(predict.unsqueeze(-1), label)
                else:
                    loss = loss_func(predict, label)
                if loss.item() < valid_loss:
                    valid_loss = loss.item()
                    output_dir = '../checkpoints/'
                    os.makedirs(output_dir, exist_ok=True)
                    path = (output_dir + args.model + '_' +
                            'feat-' + args.feat + '_' +
                            'pred_len-' + str(args.pred_len) + '_' +
                            'fold-' + str(args.fold) + '_' +
                            'node-' + str(args.pred_type) + '_' +
                            'add_feat-' + str(args.add_feat) + '_' +
                            'epoch-' + str(args.epoch) + '.pth')
                    torch.save(net.state_dict(), path)

def test(args, test_loader, occ,net,scaler='None'):
    # ----init---
    result_list = []
    predict_list = np.zeros([1, occ.shape[1], args.pred_len])
    label_list = np.zeros([1, occ.shape[1], args.pred_len])
    if args.pred_type != 'region':
        predict_list = np.zeros([1,1, args.pred_len])
        label_list = np.zeros([1,1, args.pred_len])
    # ----init---
    if not args.stat_model:
        output_dir = '../checkpoints/'
        os.makedirs(output_dir,exist_ok=True)
        path = (output_dir + args.model + '_' +
                'feat-' + args.feat + '_' +
                'pred_len-' + str(args.pred_len) + '_' +
                'fold-' + str(args.fold) + '_' +
                'node-' + str(args.pred_type) + '_' +
                'add_feat-' + str(args.add_feat) + '_' +
                'epoch-' + str(args.epoch) + '.pth')
        if os.path.exists(path):
            state_dict = torch.load(path, weights_only=True)
            net.load_state_dict(state_dict)
        else:
            print(f"[WARN] Checkpoint not found at {path}. Using current in-memory model weights.")
        net.eval()
        for j, data in enumerate(test_loader):
            extra_feat = 'None'
            if args.add_feat != 'None':
                occupancy, label, extra_feat = data
            else:
                occupancy, label = data
            with torch.no_grad():
                predict = net(occupancy, extra_feat)
                if predict.shape != label.shape:
                    predict = predict.unsqueeze(-1)
                predict = predict.cpu().detach().numpy()
            label = label.cpu().detach().numpy()

    else:
        train_valid_occ,test_occ = test_loader
        predict = net.predict(train_valid_occ,test_occ)
        horizon_len = test_occ.shape[0] - args.pred_len + 1
        label = np.stack([test_occ[i:i + args.pred_len].T for i in range(horizon_len)], axis=0)


    predict_list = np.concatenate((predict_list, predict), axis=0)
    label_list = np.concatenate((label_list, label), axis=0)
    if scaler != 'None':
        if predict_list.ndim == 3:
            reshape_predict = np.transpose(predict_list, (0, 2, 1)).reshape(-1, predict_list.shape[1])
            reshape_label = np.transpose(label_list, (0, 2, 1)).reshape(-1, label_list.shape[1])
            reshape_predict = scaler.inverse_transform(reshape_predict)
            reshape_label = scaler.inverse_transform(reshape_label)
            predict_list = np.transpose(reshape_predict.reshape(-1, args.pred_len, predict_list.shape[1]), (0, 2, 1))
            label_list = np.transpose(reshape_label.reshape(-1, args.pred_len, label_list.shape[1]), (0, 2, 1))
        else:
            predict_list = scaler.inverse_transform(predict_list)
            label_list = scaler.inverse_transform(label_list)

    eval_predict = predict_list[1:]
    eval_label = label_list[1:]
    if eval_predict.ndim == 3 and eval_predict.shape[2] == args.pred_len:
        eval_predict = np.transpose(eval_predict, (0, 2, 1))
        eval_label = np.transpose(eval_label, (0, 2, 1))

    overall_metrics, per_step_metrics = utils.metrics(test_pre=eval_predict, test_real=eval_label,args=args)
    result_list.append(overall_metrics)

    # Adding model name, pre_l and metrics and so on to DataFrame
    result_df = pd.DataFrame(result_list, columns=['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE'])
    result_df['model_name'] = args.model
    result_df['pred_len'] = args.pred_len
    result_df['fold'] = args.fold 

    # Save the results in a CSV file
    output_dir = '../result' + '/' + 'main_exp' + '/' + 'region'
    os.makedirs(output_dir, exist_ok=True)
    csv_file = output_dir + '/' + f'results.csv'
    step_csv_file = output_dir + '/' + f'results_step.csv'

    # Append the result if the file exists, otherwise create a new file
    if os.path.exists(csv_file):
        result_df.to_csv(csv_file, mode='a', header=False, index=False, encoding='gbk')
    else:
        result_df.to_csv(csv_file, index=False, encoding='gbk')

    step_result_df = pd.DataFrame(per_step_metrics, columns=['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE'])
    step_result_df['model_name'] = args.model
    step_result_df['pred_len'] = args.pred_len
    step_result_df['fold'] = args.fold
    step_result_df['step'] = np.arange(1, len(per_step_metrics) + 1)
    if os.path.exists(step_csv_file):
        step_result_df.to_csv(step_csv_file, mode='a', header=False, index=False, encoding='gbk')
    else:
        step_result_df.to_csv(step_csv_file, index=False, encoding='gbk')
