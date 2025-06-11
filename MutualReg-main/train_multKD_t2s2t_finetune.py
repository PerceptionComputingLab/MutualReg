import glob
import sys
import time
import os
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from data_utils import prepare_data, augment_affine_nl_DB
from registration_pipeline import update_fields, update_fields_multKD_DB_teach, update_fields_multKD_DB_student
from coupled_convex import coupled_convex
from natsort import natsorted
from utils_multKD import smooth_loss


def train(args):
    out_dir = args.out_dir
    smooth = 0
    flow = 10
    lr = 0.001  # 0.001
    half_iterations = 2000 * 15
    tag = '/a_t2s2t_finetune/AbdCT_t2s2t_ft_mask_lr_{}_smh_{}_flow_{}_iter{}_4.11'.format(lr, smooth, flow, half_iterations)
    out_dir = out_dir + tag
    print('tag: ', tag)
    print('Using GPU_id: ', args.gpu)

    torch.set_num_threads(4)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    num_warps = args.num_warps
    reg_fac = args.reg_fac
    use_ice = True if args.ice == 'true' else False
    use_adam = True if args.adam == 'true' else False
    do_sampling = True if args.sampling == 'true' else False
    do_augment = True if args.augment == 'true' else False

    # Loading training data (segmentations only used for validation after each stage)
    data = prepare_data(data_split='train')
    data_test = prepare_data(data_split='test')
    print()

    # loading pretrained model for Teacher
    model_dir = './results/multKD/a_teach/AbdCT_teach_lr_0.001_smh_0.1_iter20000_Tag_1.11/'
    model_idx = -2
    print('Best teacher model loaded: {}'.format(model_dir + natsorted(os.listdir(model_dir))[model_idx]))
    best_model_teacher = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx]).cuda()
    feature_net_teacher = best_model_teacher

    all_fields_teacher_adam, d_all_net_teacher, d_all0_teacher, d_all_adam_teacher, d_all_ident_teacher, \
    logJ_net_teacher, logJ_adam_teacher, Jacob_net_teacher, Jacob_adam_teacher = \
        update_fields_multKD_DB_teach(data, best_model_teacher, use_adam=True, num_warps=2, ice=True,
                                        reg_fac=10, compute_jacobian=True)
    del all_fields_teacher_adam
    print('Using Teacher model for producing "Flow GT": use_adam=True, num_warps=2, ice=True reg_fac={}'.format(
        reg_fac))
    print('Current teacher training_DSC(orig-net-adam)LogJ(net-adam)Jac(net-adam): {:.4f} {:.4f} {:.4f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(
            (d_all0_teacher.sum() / (d_all_ident_teacher > 0.1).sum()).item(),
            (d_all_net_teacher.sum() / (d_all_ident_teacher > 0.1).sum()).item(),
            (d_all_adam_teacher.sum() / (d_all_ident_teacher > 0.1).sum()).item(),
            logJ_net_teacher, logJ_adam_teacher, Jacob_net_teacher, Jacob_adam_teacher))

    # loading pretrained model for Student
    model_dir = './results/multKD/a_t2s_finetune/AbdCT_t2s_ft_mask_lr_0.001_smh_0_flow_10_iter30000_3.11/' #
    model_idx = -2
    print('Best student model loaded: {}'.format(model_dir + natsorted(os.listdir(model_dir))[model_idx]))
    best_model_student = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx]).cuda()

    all_fields_student_adam, d_all_net_student, d_all0_student, d_all_adam_student, d_all_ident_student, \
    logJ_net_student, logJ_adam_student, Jacob_net_student, Jacob_adam_student = \
        update_fields_multKD_DB_student(data, best_model_student, use_adam=True, num_warps=1, ice=True, reg_fac=3, compute_jacobian=True)
    print('Original student DSC(orig-net-adam)LogJ(net-adam)Jac(net-adam): {:.4f} {:.4f} {:.4f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(
            (d_all0_student.sum() / (d_all_ident_student > 0.1).sum()).item(),
            (d_all_net_student.sum() / (d_all_ident_student > 0.1).sum()).item(),
            (d_all_adam_student.sum() / (d_all_ident_student > 0.1).sum()).item(),
            logJ_net_student, logJ_adam_student, Jacob_net_student, Jacob_adam_student))

    N, _, H, W, D = data['images'].shape

    # #-----------------------------------------------------------------------------------------------------
    feature_net_teacher.train()
    optimizer_teacher = torch.optim.Adam(feature_net_teacher.parameters(), lr=lr)
    eta_min = 0.00001
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_teacher, 500 * 2, 1, eta_min=eta_min)
    run_lr = torch.zeros(half_iterations)
    run_loss = torch.zeros(half_iterations)
    scaler_teacher = torch.cuda.amp.GradScaler()
    from utils_multKD import MIND_loss, MSE
    loss_mind = MIND_loss()
    loss_smooth = smooth_loss
    loss_mse = MSE().loss

    # placeholders for input images, pseudo labels, and affine augmentation matrices
    img0 = torch.zeros(2, 1, H, W, D).cuda()
    img1 = torch.zeros(2, 1, H, W, D).cuda()
    target = torch.zeros(2, 3, H, W, D).cuda()
    affine1 = torch.zeros(2, H, W, D, 3).cuda()
    affine2 = torch.zeros(2, H, W, D, 3).cuda()

    t0 = time.time()
    grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(), (1, 1, H, W, D)).cuda()  # 经过归一化的grid，shape为[1, H, W, D, 3]
    best_DSC_stu_test = 0.
    best_DSC_stu_test_adam = 0.
    for i in range(half_iterations):
        optimizer_teacher.zero_grad()
        q = torch.ones(len(data['pairs']))
        idx = torch.tensor(list(WeightedRandomSampler(q, 2, replacement=True))).long()

        with torch.cuda.amp.autocast():
            # image selection and augmentation
            img0_ = data['images'][data['pairs'][idx, 0]].cuda()
            img1_ = data['images'][data['pairs'][idx, 1]].cuda()
            if do_augment:
                with torch.no_grad():
                    for j in range(len(idx)):
                        disp_field = all_fields_student_adam[idx[j]:idx[j] + 1].cuda()
                        disp_field_aff, affine1[j:j + 1], affine2[j:j + 1] = augment_affine_nl_DB(disp_field)
                        img0[j:j + 1] = F.grid_sample(img0_[j:j + 1], affine1[j:j + 1])
                        img1[j:j + 1] = F.grid_sample(img1_[j:j + 1], affine2[j:j + 1])
                        target[j:j + 1] = disp_field_aff
            else:
                with torch.no_grad():
                    for j in range(len(idx)):
                        input_field = torch.zeros((1, 3, H, W, D)).cuda()
                        disp_field_aff, affine1[j:j + 1], affine2[j:j + 1] = augment_affine_nl_DB(input_field,
                                                                                               strength=0.)
                        img0[j:j + 1] = F.grid_sample(img0_[j:j + 1], affine1[j:j + 1])
                        img1[j:j + 1] = F.grid_sample(img1_[j:j + 1], affine2[j:j + 1])
                        target[j:j + 1] = disp_field_aff
            del img0_, img1_, disp_field, disp_field_aff
            img0.requires_grad = True
            img1.requires_grad = True

            # feature extraction with feature net g
            features_fix = feature_net_teacher(img0)
            features_mov = feature_net_teacher(img1)

            # differentiable optimization with optimizer h (coupled convex)
            disp_pred = coupled_convex(features_fix, features_mov, use_ice=use_ice, img_shape=(H, W, D))

            # # #---------------------------------------------------------------------
            # # mask check
            warped_mov_pred = F.grid_sample(img1, grid0 + disp_pred.permute(0, 2, 3, 4, 1), mode='bilinear')
            _, mind_pred0 = loss_mind(warped_mov_pred[0:1], img0[0:1])
            _, mind_pred1 = loss_mind(warped_mov_pred[1:2], img0[1:2])
            mind_pred = torch.mean(torch.cat((mind_pred0, mind_pred1), dim=0), dim=1)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    warped_mov_target = F.grid_sample(img1, grid0 + target.permute(0, 2, 3, 4, 1), mode='bilinear')
                    _, mind_target0 = loss_mind(warped_mov_target.detach()[0:1].to(torch.float16), img0[0:1].to(torch.float16))
                    _, mind_target1 = loss_mind(warped_mov_target.detach()[1:2].to(torch.float16), img0[1:2].to(torch.float16))
                    mind_target = torch.mean(torch.cat((mind_target0, mind_target1), dim=0), dim=1)

                    mask_tagret = (mind_target < mind_pred)
                    mask_pred = mask_tagret.logical_not()

            def mask_for_imgORflow(img0, mask):
                img_ = torch.zeros_like(img0)
                for id, img in enumerate(img0):
                    img_mask = img * mask[id]
                    img_[id] = img_mask
                return img_

            mind_pred_mask = mask_for_imgORflow(mind_pred, mask_pred)
            loss1_sim = torch.mean(mind_pred_mask) * (mind_pred_mask.numel() / (mask_pred.sum() + 1e-5))
            loss2_smh = loss_smooth(disp_pred)

            target_mask = mask_for_imgORflow(target, mask_tagret)
            disp_pred_mask = mask_for_imgORflow(disp_pred, mask_tagret)
            loss3_flow = loss_mse(target_mask, disp_pred_mask) * (target_mask.numel() / 3 / (mask_tagret.sum() + 1e-5))

            loss = loss1_sim + loss2_smh * smooth + loss3_flow * flow
            # loss = loss1_sim + loss3_flow * flow
            # #---------------------------------------------------------------------

        scaler_teacher.scale(loss).backward()
        scaler_teacher.step(optimizer_teacher)
        scaler_teacher.update()
        scheduler.step()
        lr1 = float(scheduler.get_last_lr()[0])
        run_lr[i] = lr1

        if i % 1000 == 999:
            #  recompute pseudo-labels with current model weights
            if use_adam:
                _, d_all_net_teacher_test, d_all0_teacher_test, d_all_adam_teacher_test, d_all_ident_teacher_test, \
                logJ_net_teacher_test, logJ_adam_teacher_test, Jacob_net_teacher_test, Jacob_adam_teacher_test = \
                    update_fields_multKD_DB_teach(data_test, feature_net_teacher, use_adam=True, num_warps=2,
                                                    ice=True, reg_fac=10, compute_jacobian=True)
                del _

                print('Current teacher test_DSC(orig-net-adam)LogJ(net-adam)Jac(net-adam):{} {:.4f} {:.4f} {:.4f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(
                        i, (d_all0_teacher_test.sum() / (d_all_ident_teacher_test > 0.1).sum()).item(),
                        (d_all_net_teacher_test.sum() / (d_all_ident_teacher_test > 0.1).sum()).item(),
                        (d_all_adam_teacher_test.sum() / (d_all_ident_teacher_test > 0.1).sum()).item(),
                        logJ_net_teacher_test, logJ_adam_teacher_test, Jacob_net_teacher_test,
                        Jacob_adam_teacher_test))
                if (d_all_net_teacher_test.sum() / (d_all_ident_teacher_test > 0.1).sum()).item() > best_DSC_stu_test:
                    best_DSC_stu_test = (d_all_net_teacher_test.sum() / (d_all_ident_teacher_test > 0.1).sum()).item()
                    print('Current teacher Best test_DSC(orig-net-adam)LogJ(net-adam)Jac(net-adam):{} {:.4f} {:.4f} {:.4f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(
                            i, (d_all0_teacher_test.sum() / (d_all_ident_teacher_test > 0.1).sum()).item(),
                            (d_all_net_teacher_test.sum() / (d_all_ident_teacher_test > 0.1).sum()).item(),
                            (d_all_adam_teacher_test.sum() / (d_all_ident_teacher_test > 0.1).sum()).item(),
                            logJ_net_teacher_test, logJ_adam_teacher_test, Jacob_net_teacher_test,
                            Jacob_adam_teacher_test))
                    print(out_dir)
                if (d_all_adam_teacher_test.sum() / (d_all_ident_teacher_test > 0.1).sum()).item() > best_DSC_stu_test_adam:
                    best_DSC_stu_test_adam = (d_all_adam_teacher_test.sum() / (d_all_ident_teacher_test > 0.1).sum()).item()
                    torch.save(feature_net_teacher.cpu(), os.path.join(out_dir, 'best_DSC_stu_test_adam.pth'))

            else:
                # w/o Adam finetuning
                all_fields, d_all_net, d_all0, _, _ = update_fields(data, feature_net_teacher, use_adam=False,
                                                                    num_warps=num_warps, ice=use_ice,
                                                                    reg_fac=reg_fac)
                print('fields updated val error:', d_all0[:3].mean(), '>', d_all_net[:3].mean())

            feature_net_teacher.train()

            torch.save(feature_net_teacher.cpu(), os.path.join(
                out_dir, 'DSC{:.4f}_jacob{:.5f}_iter_'.format((d_all_net_teacher_test.sum() / (d_all_ident_teacher_test > 0.1).sum()).item(), Jacob_net_teacher_test) + str(i).zfill(5) + '.pth'))
            feature_net_teacher.cuda()
            torch.save(run_loss, os.path.join(out_dir, 'run_loss_rep.pth'))
            model_lists = natsorted(glob.glob(out_dir + '/' + '*.pth'))
            while len(model_lists) > 6:
                os.remove(model_lists[0])
                model_lists = natsorted(glob.glob(out_dir + '/' + '*.pth'))

        # # save the model for the last iter
        if i == half_iterations - 1:
            _, d_all_net_teacher_test, d_all0_teacher_test, d_all_adam_teacher_test, d_all_ident_teacher_test, \
            logJ_net_teacher_test, logJ_adam_teacher_test, Jacob_net_teacher_test, Jacob_adam_teacher_test = \
                update_fields_multKD_DB_teach(data_test, feature_net_teacher, use_adam=True, num_warps=2,
                                                ice=True, reg_fac=10, compute_jacobian=True)
            del _
            print('Final teacher test_DSC(orig-net-adam)LogJ(net-adam)Jac(net-adam):{} {:.4f} {:.4f} {:.4f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(
                    i, (d_all0_teacher_test.sum() / (d_all_ident_teacher_test > 0.1).sum()).item(),
                    (d_all_net_teacher_test.sum() / (d_all_ident_teacher_test > 0.1).sum()).item(),
                    (d_all_adam_teacher_test.sum() / (d_all_ident_teacher_test > 0.1).sum()).item(),
                    logJ_net_teacher_test, logJ_adam_teacher_test, Jacob_net_teacher_test,
                    Jacob_adam_teacher_test))
            torch.save(feature_net_teacher.cpu(), os.path.join(out_dir, 'DSC{:.4f}_jacob{:.5f}_iter_'.format(
                    (d_all_net_teacher_test.sum() / (d_all_ident_teacher_test > 0.1).sum()).item(),
                    Jacob_net_teacher_test) + str(i).zfill(5) + '_final.pth'))
            feature_net_teacher.train()

        run_loss[i] = loss.item()

        str1 = "\r" + f"iter: {i}, loss: {'%0.5f' % (run_loss[i - 34:i - 1].mean())}, loss1_sim: {'%0.5f' % (loss1_sim.item())}" \
               f", loss2_smh: {'%0.5f' % (loss2_smh.item())}, loss3_flow: {'%0.5f' % (loss3_flow.item())}, runtime: {'%0.3f' % (time.time() - t0)} sec," \
               f" GPU max/memory: {'%0.2f' % (torch.cuda.max_memory_allocated() * 1e-9)} GByte"
        sys.stdout.write(str1)
        sys.stdout.flush()

    # #----------------------------------------------------------------------------------------------------------

