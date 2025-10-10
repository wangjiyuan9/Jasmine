import sys
from collections import defaultdict

import cv2
from PIL import Image
from tqdm import tqdm

from my_utils import *
from Marigold.marigold import MarigoldPipeline
from Marigold.src.util.alignment import *

def print_errors(errors, name, type='latex'):
    if type == 'latex':
        print(("{:>20}").format(name), end='')
        print(("&{:10.3f}" * 7).format(*errors.tolist()) + "\\\\")
    elif type == 'markdown':
        print(("|{:>20}").format(name), end='')
        print(("|{:10.3f}" * 7).format(*errors.tolist()) + "|")


def print_title(name):
    print(("{:>20}").format(name), end='')
    print(("&{:>10}" * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3") + "\\\\")


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def about_stereo(opt):
    STEREO_SCALE_FACTOR = 5.4
    if opt.eval_stereo:
        print("   Stereo evaluation - disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    if opt.supervised:
        print("   Supervised evaluation - disabling median scaling, scaling by 1")
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = 1
    else:
        print("Self-Mono evaluation - using median scaling")
    return opt


def evaluate(opt, pred_disps, gt_depths, train_mode=False, train_opt=None, record=None, mode='', mono_tracker=None, global_step=None):
    errors, ratios, ratio, errors_lsq, errors_gd = [], [], 1.0, [], []
    cv2.setNumThreads(0)
    if train_mode:
        opt.eval_stereo = train_opt['eval_stereo']
        opt = about_stereo(opt)

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        if gt_depth is None:
            continue
        if opt.test_improved:
            gt_height, gt_width = gt_depth.shape[:2]
            KB_CROP_HEIGHT = 352
            KB_CROP_WIDTH = 1216
            top_margin = int(gt_height - KB_CROP_HEIGHT)
            left_margin = int((gt_width - KB_CROP_WIDTH) / 2)
            gt_depth = gt_depth[top_margin:top_margin + KB_CROP_HEIGHT, left_margin:left_margin + KB_CROP_WIDTH]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = pred_disp if opt.pred_mode == 'depth' else 1 / pred_disp
        pred_depth_org = pred_depth
        gt_depth_org = gt_depth

        if "eigen" in opt.eval_split:
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            if opt.test_improved:
                crop_mask = np.zeros(gt_depth.shape)
                crop_mask[
                int(0.3324324 * gt_height): int(0.91351351 * gt_height),
                int(0.0359477 * gt_width): int(0.96405229 * gt_width),
                ] = 1
                mask = np.logical_and(mask, crop_mask)
            else:
                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height, 0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(gt_depth.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)
        elif "city" in opt.eval_split:
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
                crop_mask = np.zeros(gt_depth.shape)
                crop_mask[256:, 192:1856] = 1
                mask = np.logical_and(mask, crop_mask)
        elif "sunny" in opt.eval_split:
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        elif "rainy" in opt.eval_split:
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        elif "cloudy" in opt.eval_split:
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        elif "foggy" in opt.eval_split:
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            
        # median scaling
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio
            
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        err = compute_errors(gt_depth, pred_depth)
        if not train_mode:
            print_errors(np.array(err), str(i) + mode + ' ' + str(ratio), type='markdown')
        errors.append(err)

        ##################### LSQ scaling   #####################
        pred_depth_org[pred_depth_org > MAX_DEPTH] = MAX_DEPTH
        pred_depth_org[pred_depth_org < MIN_DEPTH] = MIN_DEPTH
        pred_depth, scale, shift = align_depth_least_square(
            gt_arr=gt_depth_org,
            pred_arr=pred_depth_org,
            valid_mask_arr=mask,
            return_scale_shift=True,
            max_resolution=None,
        )

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth_org[mask]

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        err2 = compute_errors(gt_depth, pred_depth)
        if not train_mode:
            print_errors(np.array(err2), str(i) + mode + ' ' + str(ratio), type='markdown')
        errors_lsq.append(err2)

    return errors, errors_lsq, errors_gd


def save_depth_single(preds, inputs_val):
    paths = inputs_val["path"]
    for path, pred in zip(paths, preds):
        path = path.replace('/kitti/', '/e2e_depth/')
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        # normalize
        if pred.max() == pred.min():
            pred = np.zeros_like(pred)
        else:
            pred = (pred - pred.min()) / (pred.max() - pred.min())
        Image.fromarray((pred * 65536).astype(np.uint16)).save(path)


import multiprocessing


def validation(args, device, unet, val_dataloader, val_dataset, vae, text_encoder, val_scheduler, tokenizer, global_step, logger, accelerator, mono_tracker=None, best_abs=None, best_step=None, gt_depths=None, improvedGT=None, mode='test', inter_gru=None):
    record = defaultdict(list)
    pred_depths = torch.zeros((len(val_dataset), args.height, args.width), device=device)
    gt_3d = np.zeros((len(val_dataset), 800, 1758)) if ("sunny"in args.eval_split) or ("cloudy"in args.eval_split )or ("foggy" in args.eval_split) else np.zeros((len(val_dataset), 768, 2048))
    gt_3d = np.zeros((len(val_dataset), 800, 1762)) if "rainy" in args.eval_split else gt_3d
    pbar = tqdm(val_dataloader, desc="eval", disable=(mode == 'train'))
    start_index = 0
    inter_gru = unwrap_model(inter_gru, accelerator) if args.use_gru else None
    unet = unwrap_model(unet, accelerator)
    vae = unwrap_model(vae, accelerator)
    pipe = MarigoldPipeline(unet=unet, vae=vae, text_encoder=text_encoder, scheduler=val_scheduler, tokenizer=tokenizer, args=args,  inter_gru=inter_gru)
    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)
    pipe.unet.eval(), pipe.vae.eval()
    if args.use_gru:
        pipe.inter_gru.eval()

    with torch.no_grad():
        for idx, inputs_val in enumerate(val_dataloader):
            input_image = inputs_val["rgb_int", 0].to(device)
            end_idx = start_index + input_image.shape[0]
            if not 'eigen' in args.eval_split:
                gt_3d[start_index:end_idx] = inputs_val["depth_gt"]
            outputs_val = pipe(input_image,
                denoising_steps=args.denoise_steps,
                ensemble_size=args.ensemble_size,
                processing_res=args.processing_res,
                match_input_res='None',
                batch_size=0,
                color_map='Spectral' if idx == 0 else None,
                resample_method=args.resample_method,
            )

            pred, _ = (outputs_val.depth_tc, None) 
            if args.save_depth:
                pred = pred.cpu().numpy()
                save_depth_single(pred, inputs_val)
                pbar.update(1)
                continue

            pred_depths[start_index:end_idx] = pred.squeeze()
            pbar.update(1)
            start_index = end_idx


    for k, v in record.items():
        record[k] = torch.cat(v, 0)
    if gt_depths is not None and not args.test_improved and "eigen" in args.eval_split:
        #KITTI dataset standard evaluation (Refer to Paper Table 1)
        errors, errors_lsq, errors_gd = evaluate(args, pred_depths.cpu().numpy(), gt_depths, train_mode=(mode == 'train'), train_opt={'eval_stereo': False}, record=record, mono_tracker=mono_tracker, global_step=global_step)
        mean_errors = np.array(errors).mean(0)
        current_abs = mean_errors[0]
        for ind, error in enumerate(mean_errors):
            logger.info(f"{args.link_mode}, MED, STEP {global_step}: {index_map[ind]}: {error}")
        mean_errors_lsq = np.array(errors_lsq).mean(0)
        for ind, error in enumerate(mean_errors_lsq):
            logger.info(f"{args.link_mode}, LSQ, STEP {global_step}: {index_map[ind]}: {error}")
        accelerator.print('==================================NEW Round==================================')
    elif improvedGT is not None and args.test_improved:
        errors, errors_lsq, errors_gd = evaluate(args, pred_depths.cpu().numpy(), improvedGT, train_mode=(mode == 'train'), train_opt={'eval_stereo': False}, record=record, mono_tracker=mono_tracker, global_step=global_step)
        mean_errors = np.array(errors).mean(0)
        for ind, error in enumerate(mean_errors):
            logger.info(f"Improved, MED, STEP {global_step}: {index_map[ind]}: {error}")
        mean_errors_lsq = np.array(errors_lsq).mean(0)
        for ind, error in enumerate(mean_errors_lsq):
            logger.info(f"Improved, LSQ, STEP {global_step}: {index_map[ind]}: {error}")
    else:
        errors, errors_lsq, errors_gd = evaluate(args, pred_depths.cpu().numpy(), gt_3d, train_mode=(mode == 'train'), train_opt={'eval_stereo': False}, record=record, mono_tracker=mono_tracker, global_step=global_step)
        mean_errors = np.array(errors).mean(0)
        for ind, error in enumerate(mean_errors):
            logger.info(f"Other, MED, STEP {global_step}: {index_map[ind]}: {error}")
        mean_errors_lsq = np.array(errors_lsq).mean(0)
        for ind, error in enumerate(mean_errors_lsq):
            logger.info(f"Other, LSQ, STEP {global_step}: {index_map[ind]}: {error}")
    if mode == 'train':
        for ind, error in enumerate(mean_errors):
            mono_tracker.add_scalar('{}/{}'.format(('improved' if args.test_improved else 'first') + "中值", index_map[ind]), error, global_step)
        for ind, error in enumerate(mean_errors_lsq):
            mono_tracker.add_scalar('{}/{}'.format(('improved' if args.test_improved else 'first') + "回归", index_map[ind]), error, global_step)
        if not args.test_improved and current_abs < best_abs:
            best_abs, best_step = current_abs, global_step
        return best_abs, best_step


def prepare_save_data(fpath):
    val_filenames = readlines(fpath.format("all"))
    print("需要修改evaluate.py中的save_depth_single函数并行，还有这里的读取文件")
    partLen = len(val_filenames) // 4
    return val_filenames[partLen * 1 - 16:partLen * 1]
