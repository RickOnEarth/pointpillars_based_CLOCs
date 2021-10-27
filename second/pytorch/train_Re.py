# -*- coding: utf-8 -*-
import os
import torch
from tensorboardX import SummaryWriter
import pathlib
import pickle
import shutil
import time
from functools import partial
import sys
sys.path.append('../')
from pathlib import Path
import json
import fire
import numpy as np

from google.protobuf import text_format

import torchplus

import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.utils.eval import get_coco_eval_result, get_official_eval_result,bev_box_overlap,d3_box_overlap
from second.utils.progress_bar import ProgressBar
from second.pytorch.core import box_torch_ops
from second.pytorch.core.losses import SigmoidFocalClassificationLoss
from second.pytorch.models import fusion

def get_paddings_indicator(actual_num, max_num, axis=0):
    """
    Create boolean mask by actually number of a padded tensor.
    :param actual_num:
    :param max_num:
    :param axis:
    :return: [type]: [description]
    """
    actual_num = torch.unsqueeze(actual_num, axis+1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis+1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num : [N, M, 1]
    # tiled_actual_num : [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # title_max_num : [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape : [batch_size, max_num]
    return paddings_indicator

def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def _flat_nested_json_dict(json_dict, flatted, sep=".", start=""):
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, start + sep + k)
        else:
            flatted[start + sep + k] = v


def flat_nested_json_dict(json_dict, sep=".") -> dict:
    """flat a nested json-like dict. this function make shadow copy.
    """
    flatted = {}
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, k)
        else:
            flatted[k] = v
    return flatted


def example_convert_to_torch(example, dtype=torch.float32, device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = ["voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect", "Trv2c", "P2", "d3_gt_boxes","gt_2d_boxes"]

    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.as_tensor(v, dtype=dtype, device=device)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.as_tensor(v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.as_tensor(v, dtype=torch.uint8, device=device)
            # torch.uint8 is now deprecated, please use a dtype torch.bool instead
        else:
            example_torch[k] = v
    return example_torch


def build_inference_net(config_path,
             model_dir,
             result_path=None,
             predict_test=False,
             ckpt_path=None,
             ref_detfile=None,
             pickle_result=True,
             measure_time=False,
             batch_size=1):
    model_dir = pathlib.Path(model_dir)
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    if result_path is None:
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r", encoding="utf-8") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    model_cfg = config.model.second
    detection_2d_path = "../d2_detection_data"
    center_limit_range = model_cfg.post_center_limit_range
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    #class_names = target_assigner.classes
    net = second_builder.build(model_cfg, voxel_generator, target_assigner, batch_size)
    net.cuda()

    if ckpt_path is None:
        print("load existing model")
        torchplus.train.restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)
    batch_size = batch_size or input_cfg.batch_size
    #batch_size = 1
    net.eval()
    return net

def train(config_path,
          model_dir,
          result_path=None,
          create_folder=False,
          display_step=20,
          summary_step=5,
          pickle_result=True):
    """train a VoxelNet model specified by a config file
    """
    print("TRAIN")
    torch.manual_seed(3)            #CLOCs
    np.random.seed(3)               #CLOCs
    if create_folder:
        if pathlib.Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)

    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_checkpoint_dir = model_dir / 'eval_checkpoints'
    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r", encoding="utf-8") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    shutil.copyfile(config_path, str(model_dir / config_file_bkp))      #生成pipleline.config文件，记录当前训练的配置参数
    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    detection_2d_path = "../d2_detection_data"          #MX

    class_names = list(input_cfg.class_names)       #CLOCs: class_names = target_assigner.classes
    #########################
    # Build Voxel Generator
    #########################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    #########################
    # Build Target Assigner
    #########################
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg, bv_range, box_coder)
    center_limit_range = model_cfg.post_center_limit_range
    ######################
    # Build NetWork
    ######################
    print("build network start!")
    net = build_inference_net('./configs/pointpillars/car/CLOCs_xyres_16.proto', '../model_dir')    #MX tested
    #net = second_builder.build(model_cfg, voxel_generator, target_assigner, input_cfg.batch_size)
    net.cuda()
    print("build network end!")

    # net_train = torch.nn.DataParallel(net).cuda()
    print("num_trainable parameters:", len(list(net.parameters())))
    # for n, p in net.named_parameters():
    #     print(n, p.shape)

    """ Build fusion layer"""
    fusion_layer = fusion.fusion()
    fusion_layer.cuda()

    print("fusion layer builded")

    # # we need global_step to create lr_scheduler, so restore net first.
    # torchplus.train.try_restore_latest_checkpoints(model_dir, [net])

    ######################
    # Build Optimizer
    ######################
    gstep = net.get_global_step() - 1           #keep,MX?
    optimizer_cfg = train_cfg.optimizer
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    loss_scale = train_cfg.loss_scale_factor
    mixed_optimizer = optimizer_builder.build(optimizer_cfg, fusion_layer, mixed=train_cfg.enable_mixed_precision, loss_scale=loss_scale)
    optimizer = mixed_optimizer
    # must restore optimizer AFTER using MixedPrecisionWrapper
    torchplus.train.try_restore_latest_checkpoints(model_dir, [mixed_optimizer])

    ######################
    # Build Learning Rate Scheduler
    ######################
    #lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, optimizer, gstep)
    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, optimizer, train_cfg.steps)


    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32


    ######################
    # Prepare Input
    ######################

    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=True,                                     #MX,original为False
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)

    def _worker_init_fn(worker_id):
        time_seed = np.array(time.time(), dtype=np.int32)
        np.random.seed(time_seed + worker_id)
        print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=input_cfg.batch_size,
        shuffle=True,
        num_workers=input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_input_cfg.batch_size,
        shuffle=False,
        num_workers=eval_input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    data_iter = iter(dataloader)

    ######################
    # Training
    ######################
    focal_loss = SigmoidFocalClassificationLoss()
    cls_loss_sum = 0

    log_path = model_dir / 'log.txt'
    logf = open(log_path, 'a', encoding="utf-8")
    logf.write(proto_str)
    logf.write("\n")
    summary_dir = model_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(summary_dir))

    total_step_elapsed = 0
    remain_steps = train_cfg.steps - net.get_global_step()
    t = time.time()
    ckpt_start_time = t

    total_loop = train_cfg.steps // train_cfg.steps_per_eval + 1
    # total_loop = remain_steps // train_cfg.steps_per_eval + 1
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch

    net.set_global_step(torch.tensor([0]))                      #MX add
    if train_cfg.steps % train_cfg.steps_per_eval == 0:
        total_loop -= 1
    mixed_optimizer.zero_grad()

    try:
        for _ in range(total_loop):
            if total_step_elapsed + train_cfg.steps_per_eval > train_cfg.steps:
                steps = train_cfg.steps % train_cfg.steps_per_eval
            else:
                steps = train_cfg.steps_per_eval
            for step in range(steps):
                lr_scheduler.step(net.get_global_step())
                try:
                    example = next(data_iter)
                except StopIteration:
                    print("end epoch")
                    if clear_metrics_every_epoch:
                        net.clear_metrics()
                    data_iter = iter(dataloader)
                    example = next(data_iter)
                example_torch = example_convert_to_torch(example, float_dtype)

                # print("example_torch['image_idx']:\n", example_torch["image_idx"])

                """
                example (in evaluate()函数):
                dict_keys([0: 'voxels', 1: 'num_points', 2: 'coordinates', 3: 'rect',
                           4: 'Trv2c', 5: 'P2', 6: 'anchors', 7: 'anchors_mask', 
                           8: 'image_idx', 9: 'image_shape'])
                """

                """
                example_torch:
                [0: 'voxels', 1: 'num_points', 2: 'coordinates', 3: 'rect',
                 4: 'Trv2c', 5: 'P2', 6: 'anchors', 7: 'anchors_mask',
                 8: 'labels', 9: 'reg_targets', 10: 'reg_weights', 11: 'gt_2d_boxes',
                 12: 'd3_gt_boxes', 13: 'image_idx', 
                 14: 'image_shape']
                """
                batch_size = example["anchors"].shape[0]

                example_tuple = list(example_torch.values())
                example_tuple[13] = torch.from_numpy(example_tuple[13])
                example_tuple[14] = torch.from_numpy(example_tuple[14])

                #######################
                # get prediction_dicts
                #######################
                pillar_x = example_tuple[0][:, :, 0].unsqueeze(0).unsqueeze(0)
                pillar_y = example_tuple[0][:, :, 1].unsqueeze(0).unsqueeze(0)
                pillar_z = example_tuple[0][:, :, 2].unsqueeze(0).unsqueeze(0)
                pillar_i = example_tuple[0][:, :, 3].unsqueeze(0).unsqueeze(0)
                num_points_per_pillar = example_tuple[1].float().unsqueeze(0)

                # Find distance of x, y, and z from pillar center
                # assuming xyres_16.proto
                coors_x = example_tuple[2][:, 3].float()
                coors_y = example_tuple[2][:, 2].float()
                x_sub = coors_x.unsqueeze(1) * 0.16 + 0.1
                y_sub = coors_y.unsqueeze(1) * 0.16 + -39.9
                ones = torch.ones([1, 100], dtype=torch.float32, device=pillar_x.device)
                x_sub_shaped = torch.mm(x_sub, ones).unsqueeze(0).unsqueeze(0)
                y_sub_shaped = torch.mm(y_sub, ones).unsqueeze(0).unsqueeze(0)

                num_points_for_a_pillar = pillar_x.size()[3]
                mask = get_paddings_indicator(num_points_per_pillar, num_points_for_a_pillar, axis=0)
                mask = mask.permute(0, 2, 1)
                mask = mask.unsqueeze(1)
                mask = mask.type_as(pillar_x)

                coors = example_tuple[2]
                anchors = example_tuple[6]
                anchors_mask = example_tuple[7]
                anchors_mask = torch.as_tensor(anchors_mask, dtype=torch.uint8, device=pillar_x.device)
                anchors_mask = anchors_mask.byte()
                rect = example_tuple[3]
                Trv2c = example_tuple[4]
                P2 = example_tuple[5]
                image_idx = example_tuple[13]
                batch_image_shape = example_tuple[14]

                input = [pillar_x, pillar_y, pillar_z, pillar_i,
                         num_points_per_pillar, x_sub_shaped, y_sub_shaped,
                         mask, coors, anchors, anchors_mask, rect, Trv2c, P2, image_idx, batch_image_shape]

                all_3d_output_camera_dict, all_3d_output, top_predictions, fusion_input, tensor_index = net(input,
                                                                                                           detection_2d_path)
                d3_gt_boxes = example_torch["d3_gt_boxes"][0, :, :]


                ########################
                # 必要时，讨巧的方法，先取前70400个套用原来的代码
                #all_3d_output是voxelnet.py中的preds_dict = self.rpn(spatial_features)
                ########################

                if d3_gt_boxes.shape[0] == 0:
                    target_for_fusion = np.zeros((1,107136,1))
                    positives = torch.zeros(1,107136).type(torch.float32).cuda()
                    negatives = torch.zeros(1,107136).type(torch.float32).cuda()
                    negatives[:,:] = 1
                else:
                    d3_gt_boxes_camera = box_torch_ops.box_lidar_to_camera(
                        d3_gt_boxes, example_torch['rect'][0,:], example_torch['Trv2c'][0,:])
                    d3_gt_boxes_camera_bev = d3_gt_boxes_camera[:,[0,2,3,5,6]]
                    ###### predicted bev boxes
                    pred_3d_box = all_3d_output_camera_dict[0]["box3d_camera"]
                    pred_bev_box = pred_3d_box[:,[0,2,3,5,6]]
                    #iou_bev = bev_box_overlap(d3_gt_boxes_camera_bev.detach().cpu().numpy(), pred_bev_box.detach().cpu().numpy(), criterion=-1)
                    iou_bev = d3_box_overlap(d3_gt_boxes_camera.detach().cpu().numpy(), pred_3d_box.squeeze().detach().cpu().numpy(), criterion=-1)
                    iou_bev_max = np.amax(iou_bev,axis=0)
                    #print(np.max(iou_bev_max))
                    target_for_fusion = ((iou_bev_max >= 0.7)*1).reshape(1,-1,1)
                    target_for_fusion = target_for_fusion.astype(np.float32)
                    positive_index = ((iou_bev_max >= 0.7)*1).reshape(1,-1)
                    positive_index = positive_index.astype(np.float32)
                    positives = torch.from_numpy(positive_index).type(torch.float32).cuda()
                    negative_index = ((iou_bev_max <= 0.5)*1).reshape(1,-1)
                    negative_index = negative_index.astype(np.float32)                          ##不加这3个转换在torch中转换的话一共慢200~300ms
                    negatives = torch.from_numpy(negative_index).type(torch.float32).cuda()

                cls_preds, flag = fusion_layer(fusion_input.cuda(), tensor_index.cuda())
                one_hot_targets = torch.from_numpy(target_for_fusion).type(torch.float32).cuda()

                negative_cls_weights = negatives.type(torch.float32) * 1.0
                cls_weights = negative_cls_weights + 1.0 * positives.type(torch.float32)
                pos_normalizer = positives.sum(1, keepdim=True).type(torch.float32)
                cls_weights /= torch.clamp(pos_normalizer, min=1.0)
                if flag == 1:
                    cls_losses = focal_loss._compute_loss(cls_preds, one_hot_targets, cls_weights.cuda())  # [N, M]
                    cls_losses_reduced = cls_losses.sum() / example_torch['labels'].shape[0]
                    cls_loss_sum = cls_loss_sum + cls_losses_reduced
                    if train_cfg.enable_mixed_precision:
                        loss *= loss_scale
                    cls_losses_reduced.backward()
                    # print("cls_losses_reduced.backward(), cls_losses_reduced:\n", cls_losses_reduced)
                    mixed_optimizer.step()
                    mixed_optimizer.zero_grad()
                net.update_global_step()
                step_time = (time.time() - t)
                t = time.time()
                metrics = {}
                global_step = net.get_global_step()
                if global_step % display_step == 0:
                    print("now it is", global_step, "steps", " and the cls_loss is :", cls_loss_sum / display_step,
                          "learning_rate: ", float(optimizer.lr), file=logf)
                    print("now it is", global_step, "steps", " and the cls_loss is :", cls_loss_sum / display_step,
                          "learning_rate: ", float(optimizer.lr))
                    cls_loss_sum = 0

                ckpt_elasped_time = time.time() - ckpt_start_time

                if ckpt_elasped_time > train_cfg.save_checkpoints_secs:
                    torchplus.train.save_models(model_dir, [fusion_layer, optimizer],
                                                net.get_global_step())

                    ckpt_start_time = time.time()

                #
                # print("train, reeeeeeeeeeeeturn")
                # print("train, reeeeeeeeeeeeturn")
                # print("train, reeeeeeeeeeeeturnnn")
                # # return
                # exit()
            total_step_elapsed += steps

            torchplus.train.save_models(model_dir, [fusion_layer, optimizer],
                                        net.get_global_step())

            fusion_layer.eval()
            net.eval()
            result_path_step = result_path / f"step_{net.get_global_step()}"
            result_path_step.mkdir(parents=True, exist_ok=True)
            print("#################################")
            print("#################################", file=logf)
            print("# EVAL")
            print("# EVAL", file=logf)
            print("#################################")
            print("#################################", file=logf)
            print("Generate output labels...")
            print("Generate output labels...", file=logf)
            t = time.time()
            dt_annos = []
            prog_bar = ProgressBar()
            net.clear_timer()
            prog_bar.start((len(eval_dataset) + eval_input_cfg.batch_size - 1) // eval_input_cfg.batch_size)
            val_loss_final = 0
            for example in iter(eval_dataloader):
                example = example_convert_to_torch(example, float_dtype)
                if pickle_result:
                    dt_annos_i, val_losses = predict_kitti_to_anno(
                        net, detection_2d_path, fusion_layer, example, class_names, center_limit_range,
                        model_cfg.lidar_input)
                    dt_annos+= dt_annos_i
                    val_loss_final = val_loss_final + val_losses
                else:
                    _predict_kitti_to_file(net, detection_2d_path, example, result_path_step,
                                           class_names, center_limit_range,
                                           model_cfg.lidar_input)

                prog_bar.print_bar()

            sec_per_ex = len(eval_dataset) / (time.time() - t)
            print("validation_loss:", val_loss_final/len(eval_dataloader))
            print("validation_loss:", val_loss_final/len(eval_dataloader),file=logf)
            print(f'generate label finished({sec_per_ex:.2f}/s). start eval:')
            print(
                f'generate label finished({sec_per_ex:.2f}/s). start eval:',
                file=logf)
            gt_annos = [
                info["annos"] for info in eval_dataset.dataset.kitti_infos
            ]
            if not pickle_result:
                dt_annos = kitti.get_label_annos(result_path_step)
            # result = get_official_eval_result_v2(gt_annos, dt_annos, class_names)
            result = get_official_eval_result(gt_annos, dt_annos, class_names)
            print(result, file=logf)
            print(result)
            writer.add_text('eval_result', json.dumps(result, indent=2), global_step)
            """
            #有bug暂未解决
            result = get_coco_eval_result(gt_annos, dt_annos, class_names)
            print(result, file=logf)
            print(result)
            """
            if pickle_result:
                with open(result_path_step / "result.pkl", 'wb') as f:
                    pickle.dump(dt_annos, f)
            writer.add_text('eval_result', result, global_step)
            #net.train()
            fusion_layer.train()
    except Exception as e:

        torchplus.train.save_models(model_dir, [fusion_layer, optimizer],
                                    net.get_global_step())

        logf.close()
        raise e
    # save model before exit

    torchplus.train.save_models(model_dir, [fusion_layer, optimizer],
                                net.get_global_step())

    logf.close()


def _predict_kitti_to_file(net,
                           detection_2d_path,
                           fusion_layer,
                           example,
                           result_save_path,
                           class_names,
                           center_limit_range=None,
                           lidar_input=False):
    # eval example : [0: 'voxels', 1: 'num_points', 2: 'coordinates', 3: 'rect'
    #                 4: 'Trv2c', 5: 'P2', 6: 'anchors', 7: 'anchors_mask'
    #                 8: 'image_idx', 9: 'image_shape']

    example['image_idx'] = torch.from_numpy(example['image_idx'])
    example['image_shape'] = torch.from_numpy(example['image_shape'])
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']


    #######################
    # get prediction_dicts
    #######################
    pillar_x = example['voxels'][:, :, 0].unsqueeze(0).unsqueeze(0)
    pillar_y = example['voxels'][:, :, 1].unsqueeze(0).unsqueeze(0)
    pillar_z = example['voxels'][:, :, 2].unsqueeze(0).unsqueeze(0)
    pillar_i = example['voxels'][:, :, 3].unsqueeze(0).unsqueeze(0)
    num_points_per_pillar = example['num_points'].float().unsqueeze(0)

    # Find distance of x, y, and z from pillar center
    # assuming xyres_16.proto
    coors_x = example['coordinates'][:, 3].float()
    coors_y = example['coordinates'][:, 2].float()
    x_sub = coors_x.unsqueeze(1) * 0.16 + 0.1
    y_sub = coors_y.unsqueeze(1) * 0.16 + -39.9
    ones = torch.ones([1, 100], dtype=torch.float32, device=pillar_x.device)
    x_sub_shaped = torch.mm(x_sub, ones).unsqueeze(0).unsqueeze(0)
    y_sub_shaped = torch.mm(y_sub, ones).unsqueeze(0).unsqueeze(0)

    num_points_for_a_pillar = pillar_x.size()[3]
    mask = get_paddings_indicator(num_points_per_pillar, num_points_for_a_pillar, axis=0)
    mask = mask.permute(0, 2, 1)
    mask = mask.unsqueeze(1)
    mask = mask.type_as(pillar_x)

    coors = example['coordinates']
    anchors = example['anchors']
    anchors_mask = example['anchors_mask']
    anchors_mask = torch.as_tensor(anchors_mask, dtype=torch.uint8, device=pillar_x.device)
    anchors_mask = anchors_mask.byte()
    rect = example['rect']
    Trv2c = example['Trv2c']
    P2 = example['P2']
    image_idx = example['image_idx']

    input = [pillar_x, pillar_y, pillar_z, pillar_i,
             num_points_per_pillar, x_sub_shaped, y_sub_shaped,
             mask, coors, anchors, anchors_mask, rect, Trv2c, P2, image_idx, batch_image_shape]

    all_3d_output_camera_dict, all_3d_output, top_predictions, fusion_input, torch_index = net(input, detection_2d_path)

    #print("fusion_input.shape: \n", fusion_input.shape)         #例：[1, 4, 1, 193283]
    time_fusion_start = time.time()
    fusion_cls_preds,flag = fusion_layer(fusion_input.cuda(),torch_index.cuda())
    # print("fusion layer time: \n", (time.time() - time_fusion_start) * 1000)        #5ms 100ms 跳变

    #fusion_cls_preds_reshape = fusion_cls_preds.reshape(1,200,176,2)
    fusion_cls_preds_reshape = fusion_cls_preds.reshape(1, 248, 216, 2)

    # print("######all_3d_output:\n", all_3d_output)
    # print("######all_3d_output:\n", all_3d_output[0].shape)
    # print("######all_3d_output:\n", all_3d_output[1].shape)
    # print("######all_3d_output:\n", all_3d_output[2].shape)

    all_3d_output_dict = {}
    all_3d_output_dict["box_preds"] = all_3d_output[0]
    all_3d_output_dict["cls_preds"] = all_3d_output[1]
    all_3d_output_dict["dir_cls_preds"] = all_3d_output[2]

    predictions_dicts = predict_pp(net, example, all_3d_output_dict)

    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        image_shape = image_shape.numpy()

        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None or preds_dict["bbox"].size.numel():
            box_2d_preds = preds_dict["bbox"].data.cpu().numpy()
            box_preds = preds_dict["box3d_camera"].data.cpu().numpy()
            scores = preds_dict["scores"].data.cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].data.cpu().numpy()
            # write pred to file
            box_preds = box_preds[:, [0, 1, 2, 4, 5, 3,
                                      6]]  # lhw->hwl(label file format)
            label_preds = preds_dict["label_preds"].data.cpu().numpy()
            # print("label_preds:::\n", label_preds)
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            result_lines = []
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                # print("pred box shape:\n", box.shape)
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue

                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                result_dict = {
                    'name': class_names[int(label)],
                    'alpha': -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6],
                    'bbox': bbox,
                    'location': box[:3],
                    'dimensions': box[3:6],
                    'rotation_y': box[6],
                    'score': score,
                }
                result_line = kitti.kitti_result_line(result_dict)
                result_lines.append(result_line)
        else:
            result_lines = []
        result_file = f"{result_save_path}/{kitti.get_image_index_str(img_idx)}.txt"
        result_str = '\n'.join(result_lines)
        with open(result_file, 'w') as f:
            f.write(result_str)
        t5 = time.time()


def predict_kitti_to_anno(net,
                          detection_2d_path,
                          fusion_layer,
                          example,
                          class_names,
                          center_limit_range=None,
                          lidar_input=False,
                          global_set=None):
    time1 = time.time()
    # eval example : [0: 'voxels', 1: 'num_points', 2: 'coordinates', 3: 'rect'
    #                 4: 'Trv2c', 5: 'P2', 6: 'anchors', 7: 'anchors_mask'
    #                 8: 'image_idx', 9: 'image_shape']
    focal_loss_val = SigmoidFocalClassificationLoss()

    example['image_idx'] = torch.from_numpy(example['image_idx'])
    example['image_shape'] = torch.from_numpy(example['image_shape'])
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    #######################
    # get prediction_dicts
    #######################
    pillar_x = example['voxels'][:, :, 0].unsqueeze(0).unsqueeze(0)
    pillar_y = example['voxels'][:, :, 1].unsqueeze(0).unsqueeze(0)
    pillar_z = example['voxels'][:, :, 2].unsqueeze(0).unsqueeze(0)
    pillar_i = example['voxels'][:, :, 3].unsqueeze(0).unsqueeze(0)
    num_points_per_pillar = example['num_points'].float().unsqueeze(0)

    # Find distance of x, y, and z from pillar center
    # assuming xyres_16.proto
    coors_x = example['coordinates'][:, 3].float()
    coors_y = example['coordinates'][:, 2].float()
    x_sub = coors_x.unsqueeze(1) * 0.16 + 0.1
    y_sub = coors_y.unsqueeze(1) * 0.16 + -39.9
    ones = torch.ones([1, 100], dtype=torch.float32, device=pillar_x.device)
    x_sub_shaped = torch.mm(x_sub, ones).unsqueeze(0).unsqueeze(0)
    y_sub_shaped = torch.mm(y_sub, ones).unsqueeze(0).unsqueeze(0)

    num_points_for_a_pillar = pillar_x.size()[3]
    mask = get_paddings_indicator(num_points_per_pillar, num_points_for_a_pillar, axis=0)
    mask = mask.permute(0, 2, 1)
    mask = mask.unsqueeze(1)
    mask = mask.type_as(pillar_x)

    coors = example['coordinates']
    anchors = example['anchors']
    anchors_mask = example['anchors_mask']
    anchors_mask = torch.as_tensor(anchors_mask, dtype=torch.uint8, device=pillar_x.device)
    anchors_mask = anchors_mask.byte()
    rect = example['rect']
    Trv2c = example['Trv2c']
    P2 = example['P2']
    image_idx = example['image_idx']

    input = [pillar_x, pillar_y, pillar_z, pillar_i,
             num_points_per_pillar, x_sub_shaped, y_sub_shaped,
             mask, coors, anchors, anchors_mask, rect, Trv2c, P2, image_idx, batch_image_shape]


    all_3d_output_camera_dict, all_3d_output, top_predictions, fusion_input, torch_index = net(input, detection_2d_path)

    fusion_cls_preds,flag = fusion_layer(fusion_input.cuda(),torch_index.cuda())

    #fusion_cls_preds_reshape = fusion_cls_preds.reshape(1,200,176,2)
    fusion_cls_preds_reshape = fusion_cls_preds.reshape(1,248,216,2)

    # print("######all_3d_output:\n", all_3d_output)
    # print("######all_3d_output:\n", all_3d_output[0].shape)
    # print("######all_3d_output:\n", all_3d_output[1].shape)
    # print("######all_3d_output:\n", all_3d_output[2].shape)

    all_3d_output_dict = {}
    all_3d_output_dict["box_preds"] = all_3d_output[0]
    all_3d_output_dict["cls_preds"] = all_3d_output[1]
    all_3d_output_dict["dir_cls_preds"] = all_3d_output[2]

    predictions_dicts = predict_pp(net, example, all_3d_output_dict)
    #print("predict_kitti_to_anno net time1: ", (time.time() - time1) * 1000)
    test_mode=False
    if test_mode==False:
        d3_gt_boxes = example["d3_gt_boxes"][0,:,:]
        if d3_gt_boxes.shape[0] == 0:
            target_for_fusion = np.zeros((1,107136,1))
            positives = torch.zeros(1,107136).type(torch.float64).cuda()
            negatives = torch.zeros(1,107136).type(torch.float64).cuda()
            negatives[:,:] = 1
        else:                                                               #do

            d3_gt_boxes_camera = box_torch_ops.box_lidar_to_camera(
                d3_gt_boxes, example['rect'][0,:], example['Trv2c'][0,:])
            d3_gt_boxes_camera_bev = d3_gt_boxes_camera[:,[0,2,3,5,6]]
            ###### predicted bev boxes
            pred_3d_box = all_3d_output_camera_dict[0]["box3d_camera"]
            pred_bev_box = pred_3d_box[:,[0,2,3,5,6]]
            #iou_bev = bev_box_overlap(d3_gt_boxes_camera_bev.detach().cpu().numpy(), pred_bev_box.detach().cpu().numpy(), criterion=-1)
            iou_bev = d3_box_overlap(d3_gt_boxes_camera.detach().cpu().numpy(), pred_3d_box.squeeze().detach().cpu().numpy(), criterion=-1)
            iou_bev_max = np.amax(iou_bev,axis=0)
            time2 = time.time()
            target_for_fusion = ((iou_bev_max >= 0.7)*1).reshape(1,-1,1)
            target_for_fusion = target_for_fusion.astype(np.float32)
            positive_index = ((iou_bev_max >= 0.7)*1).reshape(1,-1)
            positive_index = positive_index.astype(np.float32)
            positives = torch.from_numpy(positive_index).type(torch.float32).cuda()
            negative_index = ((iou_bev_max <= 0.5)*1).reshape(1,-1)
            negative_index = negative_index.astype(np.float32)                              #不加这3个转换在torch中转换的话一共慢200~300ms
            negatives = torch.from_numpy(negative_index).type(torch.float32).cuda()
            #print("predict_kitti_to_anno net time2: ", (time.time() - time2) * 1000)
        cls_preds = fusion_cls_preds

        one_hot_targets = torch.from_numpy(target_for_fusion).type(torch.float32).cuda()
        #one_hot_targets = torch.as_tensor(target_for_fusion).type(torch.float32).cuda()

        negative_cls_weights = negatives.type(torch.float32) * 1.0
        cls_weights = negative_cls_weights + 1.0 * positives.type(torch.float32)
        pos_normalizer = positives.sum(1, keepdim=True).type(torch.float32)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        cls_losses = focal_loss_val._compute_loss(cls_preds, one_hot_targets, cls_weights.cuda())  # [N, M]     #20~100ms
        cls_losses_reduced = cls_losses.sum()/example['labels'].shape[0]
        cls_losses_reduced = cls_losses_reduced.detach().cpu().numpy()

    else:
        cls_losses_reduced = 1000
    # print("cls_losses_reduced:\n", cls_losses_reduced)


    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        image_shape = image_shape.numpy()
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None or preds_dict["bbox"].size.numel() != 0:
            box_2d_preds = preds_dict["bbox"].detach().cpu().numpy()
            box_preds = preds_dict["box3d_camera"].detach().cpu().numpy()
            scores = preds_dict["scores"].detach().cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].detach().cpu().numpy()
            # write pred to file
            label_preds = preds_dict["label_preds"].detach().cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            anno = kitti.get_start_result_anno()
            num_example = 0
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                anno["name"].append(class_names[int(label)])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) +
                                     box[6])
                anno["bbox"].append(bbox)
                anno["dimensions"].append(box[3:6])
                anno["location"].append(box[:3])
                anno["rotation_y"].append(box[6])
                if global_set is not None:
                    for i in range(100000):
                        if score in global_set:
                            score -= 1 / 100000
                        else:
                            global_set.add(score)
                            break
                anno["score"].append(score)

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        annos[-1]["image_idx"] = np.array(
            [img_idx] * num_example, dtype=np.int64)
        #cls_losses_reduced=100
    return annos, cls_losses_reduced


def get_prediction_dicts(net, example):
    # eval example : [0: 'voxels', 1: 'num_points', 2: 'coordinates', 3: 'rect'
    #                 4: 'Trv2c', 5: 'P2', 6: 'anchors', 7: 'anchors_mask'
    #                 8: 'image_idx', 9: 'image_shape']

    example['image_idx'] = torch.from_numpy(example['image_idx'])
    example['image_idx'] = torch.from_numpy(example['image_idx'])
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']

    pillar_x = example['voxels'][:, :, 0].unsqueeze(0).unsqueeze(0)
    pillar_y = example['voxels'][:, :, 1].unsqueeze(0).unsqueeze(0)
    pillar_z = example['voxels'][:, :, 2].unsqueeze(0).unsqueeze(0)
    pillar_i = example['voxels'][:, :, 3].unsqueeze(0).unsqueeze(0)
    num_points_per_pillar = example['num_points'].float().unsqueeze(0)

    # Find distance of x, y, and z from pillar center
    # assuming xyres_16.proto
    coors_x = example['coordinates'][:, 3].float()
    coors_y = example['coordinates'][:, 2].float()
    x_sub = coors_x.unsqueeze(1) * 0.16 + 0.1
    y_sub = coors_y.unsqueeze(1) * 0.16 + -39.9
    ones = torch.ones([1, 100], dtype=torch.float32, device=pillar_x.device)
    x_sub_shaped = torch.mm(x_sub, ones).unsqueeze(0).unsqueeze(0)
    y_sub_shaped = torch.mm(y_sub, ones).unsqueeze(0).unsqueeze(0)

    num_points_for_a_pillar = pillar_x.size()[3]
    mask = get_paddings_indicator(num_points_per_pillar, num_points_for_a_pillar, axis=0)
    mask = mask.permute(0, 2, 1)
    mask = mask.unsqueeze(1)
    mask = mask.type_as(pillar_x)

    coors   = example['coordinates']
    anchors = example['anchors']
    anchors_mask = example['anchors_mask']
    anchors_mask = torch.as_tensor(anchors_mask, dtype=torch.uint8, device=pillar_x.device)
    anchors_mask = anchors_mask.byte()
    rect = example['rect']
    Trv2c = example['Trv2c']
    P2 = example['P2']
    image_idx = example['image_idx']

    input = [pillar_x, pillar_y, pillar_z, pillar_i,
             num_points_per_pillar, x_sub_shaped, y_sub_shaped,
             mask, coors, anchors, anchors_mask, rect, Trv2c, P2, image_idx]

    predictions_dicts = net(input)

    return predictions_dicts

def evaluate(config_path,
             model_dir,
             result_path=None,
             predict_test=False,
             ckpt_path=None,
             ref_detfile=None,
             pickle_result=True):

    model_dir = str(Path(model_dir).resolve())
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    if result_path is None:
        model_dir = Path(model_dir)
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)

    if isinstance(config_path, str):
        config = pipeline_pb2.TrainEvalPipelineConfig()
        print('config_path: ', config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    detection_2d_path = "../d2_detection_data"          #MX

    class_names = list(input_cfg.class_names)
    center_limit_range = model_cfg.post_center_limit_range
    #########################
    # Build Voxel Generator
    #########################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    #########################
    # Build network
    #########################

    #net = second_builder.build(model_cfg, voxel_generator, target_assigner, input_cfg.batch_size)
    print("build network start!")
    net = build_inference_net('./configs/pointpillars/car/CLOCs_xyres_16.proto', '../model_dir')    #MX tested
    #net = second_builder.build(model_cfg, voxel_generator, target_assigner, input_cfg.batch_size)
    net.cuda()
    fusion_layer = fusion.fusion()
    fusion_layer.cuda()
    print("build network end!")
    print("fusion layer end!")

    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)

    # if ckpt_path is None:
    #     torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    # else:
    #     torchplus.train.restore(ckpt_path, net)
    ############ restore parameters for fusion layer
    if ckpt_path is None:
        print("load existing model for fusion layer")
        torchplus.train.restore_latest_checkpoints(model_dir, [fusion_layer])
        print("model_dir_pathh:\n", model_dir)
    else:
        torchplus.train.restore(ckpt_path, fusion_layer)

    print("load fusion layers done")

    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=not predict_test,              #MX, original False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=input_cfg.batch_size,
        shuffle=False,
        num_workers=input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)


    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    fusion_layer.eval()

    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    dt_annos = []
    global_set = None
    print("Generate output labels...")
    bar = ProgressBar()
    bar.start(len(eval_dataset) // input_cfg.batch_size + 1)

    print("reeeeeeeeeeeeturn")
    #return
    val_loss_final = 0  #MX
    for example in iter(eval_dataloader):
        # MX: 下面的对应关系需要验证且 Done：改成字典型
        # eval example [0: 'voxels', 1: 'num_points', 2: 'coordinates', 3: 'rect'
        #               4: 'Trv2c', 5: 'P2', 6: 'anchors', 7: 'anchors_mask'
        #               8: 'image_idx', 9: 'image_shape']
        example = example_convert_to_torch(example, float_dtype)

        """
        example: When predict_test==True 即 eval_dataset中 training=False
        dict_keys([0: 'voxels', 1: 'num_points', 2: 'coordinates', 3: 'rect',
                   4: 'Trv2c', 5: 'P2', 6: 'anchors', 7: 'anchors_mask', 
                   8: 'image_idx', 9: 'image_shape'])
        """
        """
        example: When preict_test==False 即 eval_dataset中 training=True
        [0: 'voxels', 1: 'num_points', 2: 'coordinates', 3: 'rect',
         4: 'Trv2c', 5: 'P2', 6: 'anchors', 7: 'anchors_mask',
         8: 'labels', 9: 'reg_targets', 10: 'reg_weights', 11: 'gt_2d_boxes',
         12: 'd3_gt_boxes', 13: 'image_idx', 
         14: 'image_shape']
        """

        # example_tuple = list(example.values())
        # example_tuple[8] = torch.from_numpy(example_tuple[8])
        # example_tuple[9] = torch.from_numpy(example_tuple[9])
        #

        if (example['anchors'].size()[0] != input_cfg.batch_size):
            continue

        if pickle_result:
            # dt_annos += predict_kitti_to_anno(net, example_tuple, class_names, center_limit_range,
            #     model_cfg.lidar_input, global_set)
            dt_annos_i, val_losses= predict_kitti_to_anno(
                net, detection_2d_path, fusion_layer, example, class_names, center_limit_range,
                model_cfg.lidar_input, global_set)
            dt_annos+= dt_annos_i
            val_loss_final = val_loss_final + val_losses

        else:
            #prediction_dicts = get_prediction_dicts(net, example_tuple)
            # _predict_kitti_to_file(net, example, prediction_dicts, result_path_step, class_names,
            #                        center_limit_range, model_cfg.lidar_input)
            _predict_kitti_to_file(net, detection_2d_path, fusion_layer, example, result_path_step, class_names,
                                   center_limit_range, model_cfg.lidar_input)
        bar.print_bar()

    sec_per_example = len(eval_dataset) / (time.time() - t)
    print(f'generate label finished({sec_per_example:.2f}/s). start eval:')

    #print(f"avg forward time per example: {net.avg_forward_time:.3f}")
    #print(f"avg postprocess time per example: {net.avg_postprocess_time:.3f}")
    if not predict_test:
        gt_annos = [info["annos"] for info in eval_dataset.dataset.kitti_infos]
        # if (len(gt_annos)%2 != 0):            //MX
        #     del gt_annos[-1]                  //MX
        if not pickle_result:
            dt_annos = kitti.get_label_annos(result_path_step)
        print("gt_annos:\n", len(gt_annos))
        print("dt_annos:\n", len(dt_annos))
        result = get_official_eval_result(gt_annos, dt_annos, class_names)
        print(result)
        """
        #有bug暂未解决
        result = get_coco_eval_result(gt_annos, dt_annos, class_names)
        print(result)
        """
        if pickle_result:
            with open(result_path_step / "result.pkl", 'wb') as f:
                pickle.dump(dt_annos, f)


def export_onnx(net, example, class_names, batch_image_shape,
                center_limit_range=None, lidar_input=False, global_set=None):

    pillar_x = example[0][:,:,0].unsqueeze(0).unsqueeze(0)
    pillar_y = example[0][:,:,1].unsqueeze(0).unsqueeze(0)
    pillar_z = example[0][:,:,2].unsqueeze(0).unsqueeze(0)
    pillar_i = example[0][:,:,3].unsqueeze(0).unsqueeze(0)
    num_points_per_pillar = example[1].float().unsqueeze(0)

    # Find distance of x, y, and z from pillar center
    # assuming xyres_16.proto
    coors_x = example[2][:, 3].float()
    coors_y = example[2][:, 2].float()
    x_sub = coors_x.unsqueeze(1) * 0.16 + 0.1
    y_sub = coors_y.unsqueeze(1) * 0.16 + -39.9
    ones = torch.ones([1, 100],dtype=torch.float32, device=pillar_x.device)
    x_sub_shaped = torch.mm(x_sub, ones).unsqueeze(0).unsqueeze(0)
    y_sub_shaped = torch.mm(y_sub, ones).unsqueeze(0).unsqueeze(0)

    num_points_for_a_pillar = pillar_x.size()[3]
    mask = get_paddings_indicator(num_points_per_pillar, num_points_for_a_pillar, axis=0)
    mask = mask.permute(0, 2, 1)
    mask = mask.unsqueeze(1)
    mask = mask.type_as(pillar_x)

    coors = example[2]

    print(pillar_x.size())
    print(pillar_y.size())
    print(pillar_z.size())
    print(pillar_i.size())
    print(num_points_per_pillar.size())
    print(x_sub_shaped.size())
    print(y_sub_shaped.size())
    print(mask.size())

    input_names = ["pillar_x", "pillar_y", "pillar_z", "pillar_i",
                   "num_points_per_pillar", "x_sub_shaped", "y_sub_shaped", "mask"]

    # Wierd Convloution
    pillar_x = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
    pillar_y = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
    pillar_z = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
    pillar_i = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
    num_points_per_pillar = torch.ones([1, 12000], dtype=torch.float32, device=pillar_x.device)
    x_sub_shaped = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
    y_sub_shaped = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
    mask = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)

    # De-Convolution
    # pillar_x = torch.ones([1, 100, 8599, 1],dtype=torch.float32, device=pillar_x.device )
    # pillar_y = torch.ones([1, 100, 8599, 1],dtype=torch.float32, device=pillar_x.device )
    # pillar_z = torch.ones([1, 100, 8599, 1],dtype=torch.float32, device=pillar_x.device )
    # pillar_i = torch.ones([1, 100, 8599, 1],dtype=torch.float32, device=pillar_x.device )
    # num_points_per_pillar = torch.ones([1, 8599],dtype=torch.float32, device=pillar_x.device )
    # x_sub_shaped = torch.ones([1, 100,8599, 1],dtype=torch.float32, device=pillar_x.device )
    # y_sub_shaped = torch.ones([1, 100,8599, 1],dtype=torch.float32, device=pillar_x.device )
    # mask = torch.ones([1, 100, 8599, 1],dtype=torch.float32, device=pillar_x.device )

    example1 = [pillar_x, pillar_y, pillar_z, pillar_i,
                num_points_per_pillar, x_sub_shaped, y_sub_shaped, mask]

    print('-------------- network readable visiual --------------')
    torch.onnx.export(net, example1, "pfe.onnx", verbose=False, input_names=input_names)
    print('pfe.onnx transfer success ...')

    rpn_input = torch.ones([1, 64, 496, 432], dtype=torch.float32, device=pillar_x.device)
    torch.onnx.export(net.rpn, rpn_input, "rpn.onnx", verbose=False)
    print('rpn.onnx transfer success ...')

    return 0


def onnx_model_generate(config_path,
                        model_dir,
                        result_path=None,
                        predict_test=False,
                        ckpt_path=None
                        ):
    model_dir = pathlib.Path(model_dir)
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    if result_path is None:
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    class_names = list(input_cfg.class_names)
    center_limit_range = model_cfg.post_center_limit_range

    ##########################
    ## Build Voxel Generator
    ##########################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)


    net = second_builder.build(model_cfg, voxel_generator, target_assigner, 1)
    net.cuda()
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)

    if ckpt_path is None:
        torchplus.train.restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)

    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        collate_fn=merge_second_batch)


    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)

    dt_annos = []
    global_set = None
    print("Generate output labels...")
    bar = ProgressBar()
    bar.start(len(eval_dataset) // input_cfg.batch_size + 1)

    for example in iter(eval_dataloader):
        example = example_convert_to_torch(example, float_dtype)

        example_tuple = list(example.values())
        batch_image_shape = example_tuple[8]
        example_tuple[8] = torch.from_numpy(example_tuple[8])
        example_tuple[9] = torch.from_numpy(example_tuple[9])

        dt_annos = export_onnx(
            net, example_tuple, class_names, batch_image_shape, center_limit_range,
            model_cfg.lidar_input, global_set)
        return 0
        bar.print_bar()


def predict_pp(net, example, preds_dict):
    # torch.cuda.synchronize()
    t = time.time()

    batch_size = example['anchors'].shape[0]
    batch_anchors = example['anchors'].view(batch_size, -1, 7)

    net._total_inference_count += batch_size
    batch_rect = example['rect']
    batch_Trv2c = example['Trv2c']
    batch_P2 = example['P2']
    # if "anchors_mask" not in example:
    #     batch_anchors_mask = [None] * batch_size
    # else:
    #     batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)
    #assert 15==len(example), "somthing write with example size!"
    if "anchors_mask" not in example:
        batch_anchors_mask = [None] * batch_size
        print("dasfasdafsdfasdfasdfs")
        print("dasfasdafsdfasdfasdfs")
        print("dasfasdafsdfasdfasdfs")
    else:
        batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)
    # batch_imgidx = example['image_idx']
    batch_imgidx = example['image_idx']

    #net._total_forward_time += time.time() - t
    t = time.time()
    batch_box_preds = preds_dict["box_preds"]
    batch_cls_preds = preds_dict["cls_preds"]
    batch_box_preds = batch_box_preds.view(batch_size, -1,
                                           net._box_coder.code_size)
    num_class_with_bg = net._num_class
    if not net._encode_background_as_zeros:
        num_class_with_bg = net._num_class + 1

    batch_cls_preds = batch_cls_preds.view(batch_size, -1,
                                           num_class_with_bg)
    batch_box_preds = net._box_coder.decode_torch(batch_box_preds,
                                                   batch_anchors)

    #print("batch_box_preds[0][0]:\n", batch_box_preds.shape)        #torch.Size([1, 107136, 7])
    """
    batch_box_preds[0][:20]:
    tensor([[  0.2020, -40.0929,  -1.6216,   1.5994,   3.7594,   1.4936,  -0.8914],
    [  0.2187, -39.6775,  -1.6292,   1.6008,   3.7708,   1.5095,   2.2828],
    [  0.5598, -40.0244,  -1.6149,   1.5940,   3.6686,   1.4971,  -0.8680],
    [  0.6204, -39.6814,  -1.5944,   1.5870,   3.6831,   1.5022,   2.3312],
    [  0.9631, -40.3675,  -1.7302,   1.6234,   3.7939,   1.5079,  -0.8617],
    [  1.0221, -39.8220,  -1.6777,   1.6314,   3.7000,   1.5376,   2.2164],
    [  1.2929, -40.4427,  -1.7220,   1.6193,   3.7773,   1.5189,  -0.8260],
    [  1.4031, -39.8209,  -1.6825,   1.6111,   3.9044,   1.5395,   2.2489],
    [  1.6094, -40.3091,  -1.5340,   1.5961,   4.0406,   1.5490,  -0.9331],
    [  1.6457, -39.7521,  -1.4951,   1.6042,   3.9536,   1.5647,   2.1808],
    [  1.9140, -40.3009,  -1.4752,   1.6060,   4.0838,   1.5533,  -0.9395],
    [  1.9568, -39.7552,  -1.4484,   1.5937,   3.9699,   1.5712,   2.2038],
    [  2.2469, -40.4037,  -1.2998,   1.6127,   4.1137,   1.5948,  -1.0015],
    [  2.1605, -39.6933,  -1.2740,   1.6090,   4.0216,   1.6052,   2.2176],
    [  2.5620, -40.3376,  -1.1885,   1.6066,   4.1517,   1.5938,  -1.0209],
    [  2.4918, -39.7452,  -1.1693,   1.6015,   3.9780,   1.6232,   2.2321],
    [  2.8839, -40.3407,  -0.9371,   1.6012,   4.1902,   1.5885,  -1.0024],
    [  2.8285, -39.6817,  -0.9027,   1.6110,   4.1494,   1.6197,   2.2667],
    [  3.2187, -40.3167,  -0.8476,   1.5947,   4.2698,   1.5780,  -1.0187],
    [  3.1550, -39.7382,  -0.8160,   1.6018,   4.0940,   1.6277,   2.2836]],
   device='cuda:0', grad_fn=<SliceBackward>)
    """
    if net._use_direction_classifier:
        batch_dir_preds = preds_dict["dir_cls_preds"]
        batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
    else:
        batch_dir_preds = [None] * batch_size

    predictions_dicts = []
    #predictions_dicts = ()
    for box_preds, cls_preds, dir_preds, rect, Trv2c, P2, img_idx, a_mask in zip(
            batch_box_preds, batch_cls_preds, batch_dir_preds, batch_rect,
            batch_Trv2c, batch_P2, batch_imgidx, batch_anchors_mask
    ):
        if a_mask is not None:
            box_preds = box_preds[a_mask]
            cls_preds = cls_preds[a_mask]

        if net._use_direction_classifier:
            if a_mask is not None:
                dir_preds = dir_preds[a_mask]
            # print(dir_preds.shape)
            dir_labels = torch.max(dir_preds, dim=-1)[1]
        if net._encode_background_as_zeros:
            # this don't support softmax
            assert net._use_sigmoid_score is True
            total_scores = torch.sigmoid(cls_preds)
        else:
            # encode background as first element in one-hot vector
            if net._use_sigmoid_score:
                total_scores = torch.sigmoid(cls_preds)[..., 1:]
            else:
                total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
        # Apply NMS in birdeye view
        if net._use_rotate_nms:
            nms_func = box_torch_ops.rotate_nms
        else:
            nms_func = box_torch_ops.nms
        selected_boxes = None
        selected_labels = None
        selected_scores = None
        selected_dir_labels = None

        if net._multiclass_nms:
            # curently only support class-agnostic boxes.
            boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
            if not net._use_rotate_nms:
                box_preds_corners = box_torch_ops.center_to_corner_box2d(
                    boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                    boxes_for_nms[:, 4])
                boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                    box_preds_corners)
            boxes_for_mcnms = boxes_for_nms.unsqueeze(1)
            selected_per_class = box_torch_ops.multiclass_nms(
                nms_func=nms_func,
                boxes=boxes_for_mcnms,
                scores=total_scores,
                num_class=net._num_class,
                pre_max_size=net._nms_pre_max_size,
                post_max_size=net._nms_post_max_size,
                iou_threshold=net._nms_iou_threshold,
                score_thresh=net._nms_score_threshold,
            )
            selected_boxes, selected_labels, selected_scores = [], [], []
            selected_dir_labels = []
            for i, selected in enumerate(selected_per_class):
                if selected is not None:
                    num_dets = selected.shape[0]
                    selected_boxes.append(box_preds[selected])
                    selected_labels.append(
                        torch.full([num_dets], i, dtype=torch.int64))
                    if net._use_direction_classifier:
                        selected_dir_labels.append(dir_labels[selected])
                    selected_scores.append(total_scores[selected, i])
            if len(selected_boxes) > 0:
                selected_boxes = torch.cat(selected_boxes, dim=0)
                selected_labels = torch.cat(selected_labels, dim=0)
                selected_scores = torch.cat(selected_scores, dim=0)
                if net._use_direction_classifier:
                    selected_dir_labels = torch.cat(
                        selected_dir_labels, dim=0)
            else:
                selected_boxes = None
                selected_labels = None
                selected_scores = None
                selected_dir_labels = None
        else:
            # get highest score per prediction, than apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(-1)
                top_labels = torch.zeros(
                    total_scores.shape[0],
                    device=total_scores.device,
                    dtype=torch.long)
            else:
                top_scores, top_labels = torch.max(total_scores, dim=-1)

            if net._nms_score_threshold > 0.0:
                thresh = torch.tensor(
                    [net._nms_score_threshold],           #MX!MX: [0.5],
                    device=total_scores.device).type_as(total_scores)
                top_scores_keep = (top_scores >= thresh)
                top_scores = top_scores.masked_select(top_scores_keep)
            if top_scores.shape[0] != 0:
                if net._nms_score_threshold > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    if net._use_direction_classifier:
                        dir_labels = dir_labels[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not net._use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners)
                # the nms in 3d detection just remove overlap boxes.
                selected = nms_func(
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=net._nms_pre_max_size,
                    post_max_size=net._nms_post_max_size,
                    iou_threshold=net._nms_iou_threshold,         #MX!MX: iou_threshold=0.5
                )
            else:
                selected = None
            if selected is not None:
                selected_boxes = box_preds[selected]
                if net._use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
        # finally generate predictions.

        if selected_boxes is not None:
            box_preds = selected_boxes
            scores = selected_scores
            label_preds = selected_labels
            if net._use_direction_classifier:
                dir_labels = selected_dir_labels
                opp_labels = (box_preds[..., -1] > 0) ^ (dir_labels.byte() > 0)
                box_preds[..., -1] += torch.where(
                    opp_labels,
                    torch.tensor(np.pi).type_as(box_preds),
                    torch.tensor(0.0).type_as(box_preds))
                # box_preds[..., -1] += (
                #     ~(dir_labels.byte())).type_as(box_preds) * np.pi
            final_box_preds = box_preds
            final_scores = scores
            final_labels = label_preds
            final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                final_box_preds, rect, Trv2c)
            locs = final_box_preds_camera[:, :3]
            dims = final_box_preds_camera[:, 3:6]
            angles = final_box_preds_camera[:, 6]
            camera_box_origin = [0.5, 1.0, 0.5]
            box_corners = box_torch_ops.center_to_corner_box3d(
                locs, dims, angles, camera_box_origin, axis=1)

            box_corners_in_image = box_torch_ops.project_to_image(
                box_corners, P2)

            # box_corners_in_image: [N, 8, 2]
            minxy = torch.min(box_corners_in_image, dim=1)[0]
            maxxy = torch.max(box_corners_in_image, dim=1)[0]
            # minx = torch.min(box_corners_in_image[..., 0], dim=1)[0]
            # maxx = torch.max(box_corners_in_image[..., 0], dim=1)[0]
            # miny = torch.min(box_corners_in_image[..., 1], dim=1)[0]
            # maxy = torch.max(box_corners_in_image[..., 1], dim=1)[0]
            # box_2d_preds = torch.stack([minx, miny, maxx, maxy], dim=1)
            box_2d_preds = torch.cat([minxy, maxxy], dim=1)

            # predictions
            predictions_dict = {
                "bbox": box_2d_preds,
                "box3d_camera": final_box_preds_camera,
                "box3d_lidar": final_box_preds,
                "scores": final_scores,
                "label_preds": label_preds,
                "image_idx": img_idx,
            }
        else:
            dtype = batch_box_preds.dtype
            device = batch_box_preds.device
            predictions_dict = {
                "bbox": torch.zeros([0, 4], dtype=dtype, device=device),
                "box3d_camera": torch.zeros([0, 7], dtype=dtype, device=device),
                "box3d_lidar": torch.zeros([0, 7], dtype=dtype, device=device),
                "scores": torch.zeros([0], dtype=dtype, device=device),
                "label_preds": torch.zeros([0, 4], dtype=top_labels.dtype, device=device),
                "image_idx": img_idx,
            }
        predictions_dicts.append(predictions_dict)
        #predictions_dicts += (predictions_dict, )
    #net._total_postprocess_time += time.time() - t
    return predictions_dicts


def predict_v2(net,example, preds_dict):
    # eval example : [0: 'voxels', 1: 'num_points', 2: 'coordinates', 3: 'rect'
    #                 4: 'Trv2c', 5: 'P2', 6: 'anchors', 7: 'anchors_mask'
    #                 8: 'image_idx', 9: 'image_shape']

    t = time.time()
    batch_size = example['anchors'].shape[0]
    batch_anchors = example['anchors'].view(batch_size, -1, 7)

    ########################
    # 讨巧的方法，先取前70400个anchor套用原来的代码
    ########################
    #batch_anchors = batch_anchors[:, :70400, :]
    #exit()!!!  batch_anchors等的shape调整70400

    batch_rect = example['rect']
    batch_Trv2c = example['Trv2c']
    batch_P2 = example['P2']
    # if "anchors_mask" not in example:
    #     batch_anchors_mask = [None] * batch_size
    # else:
    #     batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)
    batch_anchors_mask = [None] * batch_size        #MX

    batch_imgidx = example['image_idx']

    t = time.time()
    batch_box_preds = preds_dict["box_preds"]
    batch_cls_preds = preds_dict["cls_preds"]
    batch_box_preds = batch_box_preds.view(batch_size, -1,
                                           net._box_coder.code_size)

    ########################
    # 讨巧的方法，先取前70400个套用原来的代码
    ########################
    #batch_box_preds = batch_box_preds[:, :70400, :]  # shape: [1, 70400, 7]

    num_class_with_bg = net._num_class
    if not net._encode_background_as_zeros:
        num_class_with_bg = net._num_class + 1
    batch_cls_preds = batch_cls_preds.view(batch_size, -1,
                                           num_class_with_bg)

    ########################
    # 讨巧的方法，先取前70400个套用原来的代码
    ########################
    #batch_cls_preds = batch_cls_preds[:, :70400, :]  # batch_cls_preds.shape:[1, 70400, 1]


    batch_box_preds = net._box_coder.decode_torch(batch_box_preds,
                                                   batch_anchors)
    if net._use_direction_classifier:
        batch_dir_preds = preds_dict["dir_cls_preds"]
        batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)

        ########################
        # 讨巧的方法，先取前70400个套用原来的代码
        ########################
        #batch_dir_preds = batch_dir_preds[:, :70400, :]

    else:
        batch_dir_preds = [None] * batch_size

    predictions_dicts = []
    for box_preds, cls_preds, dir_preds, rect, Trv2c, P2, img_idx, a_mask in zip(
            batch_box_preds, batch_cls_preds, batch_dir_preds, batch_rect,
            batch_Trv2c, batch_P2, batch_imgidx, batch_anchors_mask):
        if a_mask is not None:
            box_preds = box_preds[a_mask]
            cls_preds = cls_preds[a_mask]
        box_preds = box_preds.float()
        cls_preds = cls_preds.float()
        rect = rect.float()
        Trv2c = Trv2c.float()
        P2 = P2.float()
        if net._use_direction_classifier:
            if a_mask is not None:
                dir_preds = dir_preds[a_mask]
            dir_labels = torch.max(dir_preds, dim=-1)[1]
        if net._encode_background_as_zeros:
            # this don't support softmax
            assert net._use_sigmoid_score is True
            total_scores = torch.sigmoid(cls_preds)
        else:
            # encode background as first element in one-hot vector
            if net._use_sigmoid_score:
                total_scores = torch.sigmoid(cls_preds)[..., 1:]
            else:
                total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
        # Apply NMS in birdeye view
        if net._use_rotate_nms:
            nms_func = box_torch_ops.rotate_nms
        else:
            nms_func = box_torch_ops.nms

        if net._multiclass_nms:
            # curently only support class-agnostic boxes.
            boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
            if not net._use_rotate_nms:
                box_preds_corners = box_torch_ops.center_to_corner_box2d(
                    boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                    boxes_for_nms[:, 4])
                boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                    box_preds_corners)
            boxes_for_mcnms = boxes_for_nms.unsqueeze(1)
            selected_per_class = box_torch_ops.multiclass_nms(
                nms_func=nms_func,
                boxes=boxes_for_mcnms,
                scores=total_scores,
                num_class=net._num_class,
                pre_max_size=net._nms_pre_max_size,
                post_max_size=net._nms_post_max_size,
                iou_threshold=net._nms_iou_threshold,
                score_thresh=net._nms_score_threshold,
            )
            selected_boxes, selected_labels, selected_scores = [], [], []
            selected_dir_labels = []
            for i, selected in enumerate(selected_per_class):
                if selected is not None:
                    num_dets = selected.shape[0]
                    selected_boxes.append(box_preds[selected])
                    selected_labels.append(
                        torch.full([num_dets], i, dtype=torch.int64))
                    if net._use_direction_classifier:
                        selected_dir_labels.append(dir_labels[selected])
                    selected_scores.append(total_scores[selected, i])
            selected_boxes = torch.cat(selected_boxes, dim=0)
            selected_labels = torch.cat(selected_labels, dim=0)
            selected_scores = torch.cat(selected_scores, dim=0)
            if net._use_direction_classifier:
                selected_dir_labels = torch.cat(
                    selected_dir_labels, dim=0)
        else:
            # get highest score per prediction, than apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(-1)
                top_labels = torch.zeros(
                    total_scores.shape[0],
                    device=total_scores.device,
                    dtype=torch.long)
            else:
                top_scores, top_labels = torch.max(total_scores, dim=-1)

            if net._nms_score_threshold > 0.0:
                thresh = torch.tensor(
                    [net._nms_score_threshold],
                    device=total_scores.device).type_as(total_scores)
                top_scores_keep = (top_scores >= thresh)
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if net._nms_score_threshold > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    if net._use_direction_classifier:
                        dir_labels = dir_labels[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not net._use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners)
                # the nms in 3d detection just remove overlap boxes.
                selected = nms_func(
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=net._nms_pre_max_size,
                    post_max_size=net._nms_post_max_size,
                    iou_threshold=net._nms_iou_threshold,
                )

            else:
                selected = []
            # if selected is not None:
            selected_boxes = box_preds[selected]
            if net._use_direction_classifier:
                selected_dir_labels = dir_labels[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]
        # finally generate predictions.
        if selected_boxes.shape[0] != 0:
            box_preds = selected_boxes
            scores = selected_scores
            label_preds = selected_labels
            if net._use_direction_classifier:
                dir_labels = selected_dir_labels
                #print("dir_labels shape is:",dir_labels.shape,"the values are: ",dir_labels)
                opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.to(torch.bool)
                box_preds[..., -1] += torch.where(
                    opp_labels,
                    torch.tensor(np.pi).type_as(box_preds),
                    torch.tensor(0.0).type_as(box_preds))
            final_box_preds = box_preds
            final_scores = scores
            final_labels = label_preds
            final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                final_box_preds, rect, Trv2c)
            locs = final_box_preds_camera[:, :3]
            dims = final_box_preds_camera[:, 3:6]
            angles = final_box_preds_camera[:, 6]
            camera_box_origin = [0.5, 1.0, 0.5]
            box_corners = box_torch_ops.center_to_corner_box3d(
                locs, dims, angles, camera_box_origin, axis=1)

            box_corners_in_image = box_torch_ops.project_to_image(
                box_corners, P2)
            # box_corners_in_image: [N, 8, 2]
            minxy = torch.min(box_corners_in_image, dim=1)[0]
            maxxy = torch.max(box_corners_in_image, dim=1)[0]
            box_2d_preds = torch.cat([minxy, maxxy], dim=1)
            # predictions
            predictions_dict = {
                "bbox": box_2d_preds,
                "box3d_camera": final_box_preds_camera,
                "box3d_lidar": final_box_preds,
                "scores": final_scores,
                "label_preds": label_preds,
                "image_idx": img_idx,
            }
        else:
            dtype = batch_box_preds.dtype
            device = batch_box_preds.device
            predictions_dict = {
                "bbox": torch.zeros([0, 4], dtype=dtype, device=device),
                "box3d_camera": torch.zeros([0, 7], dtype=dtype, device=device),
                "box3d_lidar": torch.zeros([0, 7], dtype=dtype, device=device),
                "scores": torch.zeros([0], dtype=dtype, device=device),
                "label_preds": torch.zeros([0, 4], dtype=top_labels.dtype, device=device),
                "image_idx": img_idx,
            }
        predictions_dicts.append(predictions_dict)
    return predictions_dicts


if __name__ == '__main__':
    print("mian")
    fire.Fire()
