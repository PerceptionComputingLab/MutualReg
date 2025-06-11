import glob
import sys
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from data_utils import prepare_data, augment_affine_nl
from registration_pipeline import update_fields, update_fields_multKD_DB_student
from coupled_convex import coupled_convex
from natsort import natsorted
from utils_multKD import smooth_loss


def train(args):
    out_dir = args.out_dir
    smooth = 0.1
    lr = 0.001  # 0.001
    half_iterations = 2000 * 10
    tag = '/a_student/AbdCT_student_lr_{}_smh_{}_iter{}_Tag_2.11'.format(lr, smooth, half_iterations)
    out_dir = out_dir + tag
    print('tag: ', tag)
    torch.set_num_threads(4)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('Using GPU_id: ', args.gpu)
    num_warps = args.num_warps
    reg_fac = args.reg_fac
    use_ice = True if args.ice == 'true' else False
    use_adam = True if args.adam == 'true' else False
    do_sampling = True if args.sampling == 'true' else False
    do_augment = True if args.augment == 'true' else False

    # Loading training data (segmentations only used for validation after each stage)
    data = prepare_data(data_split='train')
    data_test = prepare_data(data_split='test')

    # initialize feature net
    feature_net_student = nn.Sequential(nn.Conv3d(1, 32, 3, padding=1, stride=2), nn.BatchNorm3d(32), nn.ReLU(),
                                        nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
                                        nn.Conv3d(64, 128, 3, padding=1, stride=2), nn.BatchNorm3d(128), nn.ReLU(),
                                        nn.Conv3d(128, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
                                        nn.Conv3d(128, 128, 3, padding=1, stride=2), nn.BatchNorm3d(128), nn.ReLU(),
                                        nn.Conv3d(128, 16, 1)).cuda()
    print()

    N, _, H, W, D = data['images'].shape

    # #-----------------------------------------------------------------------------------------------------
    feature_net_student.train()
    optimizer_student = torch.optim.Adam(feature_net_student.parameters(), lr=lr)
    eta_min = 0.00001
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_student, half_iterations+2, 1, eta_min=eta_min)
    run_lr = torch.zeros(half_iterations)
    run_loss = torch.zeros(half_iterations)
    scaler_student = torch.cuda.amp.GradScaler()
    from utils_multKD import MIND_loss
    loss_mind = MIND_loss()
    loss_smooth = smooth_loss

    # placeholders for input images, pseudo labels, and affine augmentation matrices
    img0 = torch.zeros(2, 1, H, W, D).cuda()
    img1 = torch.zeros(2, 1, H, W, D).cuda()
    target = torch.zeros(2, 3, H // 2, W // 2, D // 2).cuda()
    affine1 = torch.zeros(2, H, W, D, 3).cuda()
    affine2 = torch.zeros(2, H, W, D, 3).cuda()

    t0 = time.time()
    grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(), (1, 1, H, W, D)).cuda()  # 经过归一化的grid，shape为[1, H, W, D, 3]
    best_DSC_stu_test = 0.
    # with tqdm(total=half_iterations, file=sys.stdout, colour="red") as pbar:
    for i in range(half_iterations):
        optimizer_student.zero_grad()
        # difficulty weighting
        q = torch.ones(len(data['pairs']))
        idx = torch.tensor(list(WeightedRandomSampler(q, 2, replacement=True))).long()

        with torch.cuda.amp.autocast():
            # image selection and augmentation
            img0_ = data['images'][data['pairs'][idx, 0]].cuda()
            img1_ = data['images'][data['pairs'][idx, 1]].cuda()
            if do_augment:
                with torch.no_grad():
                    for j in range(len(idx)):
                        disp_field = torch.zeros((1, 3, H, W, D)).cuda()
                        disp_field_aff, affine1[j:j + 1], affine2[j:j + 1] = augment_affine_nl(disp_field)
                        img0[j:j + 1] = F.grid_sample(img0_[j:j + 1], affine1[j:j + 1])
                        img1[j:j + 1] = F.grid_sample(img1_[j:j + 1], affine2[j:j + 1])
                        target[j:j + 1] = disp_field_aff
            else:
                with torch.no_grad():
                    for j in range(len(idx)):
                        input_field = torch.zeros((1, 3, H, W, D)).cuda()
                        disp_field_aff, affine1[j:j + 1], affine2[j:j + 1] = augment_affine_nl(input_field,
                                                                                               strength=0.)
                        img0[j:j + 1] = F.grid_sample(img0_[j:j + 1], affine1[j:j + 1])
                        img1[j:j + 1] = F.grid_sample(img1_[j:j + 1], affine2[j:j + 1])
                        target[j:j + 1] = disp_field_aff
            img0.requires_grad = True
            img1.requires_grad = True

            # feature extraction with feature net g
            features_fix = feature_net_student(img0)
            features_mov = feature_net_student(img1)

            # differentiable optimization with optimizer h (coupled convex)
            disp_pred = coupled_convex(features_fix, features_mov, use_ice=use_ice, img_shape=(H, W, D))

            warped_mov = F.grid_sample(img1, grid0 + disp_pred.permute(0, 2, 3, 4, 1), mode='bilinear')
            loss1_sim, _ = loss_mind(warped_mov, img0)
            loss2_smh = loss_smooth(disp_pred)
            loss = loss1_sim + loss2_smh * smooth

        scaler_student.scale(loss).backward()
        scaler_student.step(optimizer_student)
        scaler_student.update()
        scheduler.step()
        lr1 = float(scheduler.get_last_lr()[0])
        run_lr[i] = lr1

        if i % 1000 == 999:
            #  recompute pseudo-labels with current model weights
            if use_adam:
                _, d_all_net_student_test, d_all0_student_test, d_all_adam_student_test, d_all_ident_student_test, \
                logJ_net_student_test, logJ_adam_student_test, Jacob_net_student_test, Jacob_adam_student_test = \
                    update_fields_multKD_DB_student(data_test, feature_net_student, use_adam=True, num_warps=2,
                                                    ice=True, reg_fac=10, compute_jacobian=True)

                # # recompute difference between finetuned and non-finetuned fields for difficulty sampling --> the larger the difference, the more difficult the sample
                print('Current student test_DSC(orig-net-adam)LogJ(net-adam)Jac(net-adam):{} {:.4f} {:.4f} {:.4f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(
                    i, (d_all0_student_test.sum() / (d_all_ident_student_test > 0.1).sum()).item(),
                    (d_all_net_student_test.sum() / (d_all_ident_student_test > 0.1).sum()).item(),
                    (d_all_adam_student_test.sum() / (d_all_ident_student_test > 0.1).sum()).item(),
                    logJ_net_student_test, logJ_adam_student_test, Jacob_net_student_test,
                    Jacob_adam_student_test))
                if (d_all_net_student_test.sum() / (d_all_ident_student_test > 0.1).sum()).item() > best_DSC_stu_test:
                    best_DSC_stu_test = (d_all_net_student_test.sum() / (d_all_ident_student_test > 0.1).sum()).item()
                    print('Current student Best test_DSC(orig-net-adam)LogJ(net-adam)Jac(net-adam):{} {:.4f} {:.4f} {:.4f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(
                            i, (d_all0_student_test.sum() / (d_all_ident_student_test > 0.1).sum()).item(),
                            (d_all_net_student_test.sum() / (d_all_ident_student_test > 0.1).sum()).item(),
                            (d_all_adam_student_test.sum() / (d_all_ident_student_test > 0.1).sum()).item(),
                            logJ_net_student_test, logJ_adam_student_test, Jacob_net_student_test,
                            Jacob_adam_student_test))
                    print(out_dir)

            else:
                # w/o Adam finetuning
                all_fields, d_all_net, d_all0, _, _ = update_fields(data, feature_net_student, use_adam=False,
                                                                    num_warps=num_warps, ice=use_ice,
                                                                    reg_fac=reg_fac)
                print('fields updated val error:', d_all0[:3].mean(), '>', d_all_net[:3].mean())

            feature_net_student.train()

            torch.save(feature_net_student.cpu(), os.path.join(
                out_dir, 'DSC{:.4f}_jacob{:.5f}_iter_'.format((d_all_net_student_test.sum() / (d_all_ident_student_test > 0.1).sum()).item(), Jacob_net_student_test) + str(i).zfill(5) + '.pth'))
            feature_net_student.cuda()
            torch.save(run_loss, os.path.join(out_dir, 'run_loss_rep.pth'))
            model_lists = natsorted(glob.glob(out_dir + '/' + '*.pth'))
            while len(model_lists) > 5:
                os.remove(model_lists[0])
                model_lists = natsorted(glob.glob(out_dir + '/' + '*.pth'))

        run_loss[i] = loss.item()

        str1 = "\r" + f"iter: {i}, loss: {'%0.5f' % (run_loss[i - 34:i - 1].mean())}, loss1_sim: {'%0.5f' % (loss1_sim.item())}" \
               f", loss2_smh: {'%0.5f' % (loss2_smh.item())}, runtime: {'%0.3f' % (time.time() - t0)} sec," \
               f" GPU max/memory: {'%0.2f' % (torch.cuda.max_memory_allocated() * 1e-9)} GByte"
        sys.stdout.write(str1)
        sys.stdout.flush()

    # #----------------------------------------------------------------------------------------------------------





