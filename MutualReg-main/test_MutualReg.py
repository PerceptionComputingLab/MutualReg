import torch
from data_utils import prepare_data
from registration_pipeline import update_fields, update_fields_multKD,\
    update_fields_multKD_DB_teach, update_fields_multKD_DB_student
import os
import numpy as np


def test():
    data = prepare_data(data_split='test')
    model = './results/multKD/a_teach/AbdCT_teach_lr_0.001_smh_0.1_iter50000_Tag_1.4.11/' \
            'DSC0.4284_jacob0.41245_iter_43999.pth'
    print('-'*70)
    print(model)
    print('12345')
    reg_fac = 15
    print('reg_fac: ', reg_fac)
    print('-'*70)
    feature_net = torch.load(model).cuda()


    # all_fields, d_all_net, d_all0, d_all_adam, d_all_ident = update_fields(data, feature_net, True, num_warps=2, compute_jacobian=True, ice=False, reg_fac=10.)
    all_fields_test, d_all_net_warp1, d_all_net_warp2, d_all0_test, d_all_adam_test, d_all_ident_test, logJ_net_test, logJ_adam_test, Jacob_net_test, Jacob_adam_test = update_fields_multKD_DB_teach(
        data, feature_net, True, num_warps=2, compute_jacobian=True, ice=True, reg_fac=reg_fac)

    # all_fields, d_all_net, d_all0, d_all_adam, d_all_ident,_,_,_,_ = update_fields_multKD_diff(data, feature_net, True, num_warps=2, compute_jacobian=True, ice=False, reg_fac=10.)
    # print('DSC:', (d_all0.sum() / (d_all_ident > 0.1).sum()).item(), '>', (d_all_net.sum() / (d_all_ident > 0.1).sum()).item(), (d_all_adam.sum() / (d_all_ident > 0.1).sum()).item())
    # print('DSC(orig-net-adam): {:.4f} {:.4f} {:.4f}'.format((d_all0.sum() / (d_all_ident > 0.1).sum()).item(), (d_all_net.sum() / (d_all_ident > 0.1).sum()).item(), (d_all_adam.sum() / (d_all_ident > 0.1).sum()).item()))
    print('Current best DSC(raw-warp1-warp2-adam)LogJ(net-adam)Jac(net-adam): {:.4f} {:.4f} {:.4f} {:.4f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(
        (d_all0_test.sum() / (d_all_ident_test > 0.1).sum()).item(),
        (d_all_net_warp1.sum() / (d_all_ident_test > 0.1).sum()).item(),
        (d_all_net_warp2.sum() / (d_all_ident_test > 0.1).sum()).item(),
        (d_all_adam_test.sum() / (d_all_ident_test > 0.1).sum()).item(),
        logJ_net_test, logJ_adam_test, Jacob_net_test, Jacob_adam_test))

    print(all_fields_test.shape)

    # flow_save_dir = './flow_save_dir/test_npy/'
    # if not os.path.exists(flow_save_dir):
    #     os.makedirs(flow_save_dir)
    # # from utils_multKD import write_pickle
    # for i, datas in enumerate(all_fields_test):
    #     # write_pickle(all_fields_test[i:i+1].data, flow_save_dir + f'{i}.pkl')
    #     np.save(flow_save_dir + f'{i}.npy', all_fields_test[i:i+1].data)




