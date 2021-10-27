### def predict_kitti_to_anno()
predic_kitti_to_anno 与 _predict_kitti_to_file()相比多了val_losses的计算与返回  
问题：loss的计算方式要与pointpillars对应吧？
```python
    test_mode=False
    if test_mode==False:
        d3_gt_boxes = example["d3_gt_boxes"][0,:,:]
        if d3_gt_boxes.shape[0] == 0:
            target_for_fusion = np.zeros((1,107136,1))
            positives = torch.zeros(1,107136).type(torch.float32).cuda()
            negatives = torch.zeros(1,107136).type(torch.float32).cuda()
            negatives[:,:] = 1
        else:                                                               #do
            time1 = time.time()
            d3_gt_boxes_camera = box_torch_ops.box_lidar_to_camera(
                d3_gt_boxes, example['rect'][0,:], example['Trv2c'][0,:])
            d3_gt_boxes_camera_bev = d3_gt_boxes_camera[:,[0,2,3,5,6]]
            ###### predicted bev boxes
            pred_3d_box = all_3d_output_camera_dict[0]["box3d_camera"]
            pred_bev_box = pred_3d_box[:,[0,2,3,5,6]]
            #iou_bev = bev_box_overlap(d3_gt_boxes_camera_bev.detach().cpu().numpy(), pred_bev_box.detach().cpu().numpy(), criterion=-1)
            iou_bev = d3_box_overlap(d3_gt_boxes_camera.detach().cpu().numpy(), pred_3d_box.squeeze().detach().cpu().numpy(), criterion=-1)
            iou_bev_max = np.amax(iou_bev,axis=0)
            target_for_fusion = ((iou_bev_max >= 0.7)*1).reshape(1,-1,1)
            positive_index = ((iou_bev_max >= 0.7)*1).reshape(1,-1)
            positives = torch.from_numpy(positive_index).type(torch.float32).cuda()
            negative_index = ((iou_bev_max <= 0.5)*1).reshape(1,-1)
            negatives = torch.from_numpy(negative_index).type(torch.float32).cuda()
            print("predict_kitti_to_anno net time1: ", (time.time() - time1) * 1000)

        cls_preds = fusion_cls_preds
        time2 = time.time()
        print("target_for_fusion \n", target_for_fusion)
        print("target_for_fusion.shape \n", target_for_fusion.shape)
        print("max \n", max(max(max(target_for_fusion))))
        one_hot_targets = torch.from_numpy(target_for_fusion).type(torch.float32).cuda()
        #one_hot_targets = torch.as_tensor(target_for_fusion).type(torch.float32).cuda()
        print("one_hot_targets \n", one_hot_targets)
        print("one_hot_targets.shape \n", one_hot_targets.shape)
        print("predict_kitti_to_anno net time2: ", (time.time() - time2) * 1000)
        negative_cls_weights = negatives.type(torch.float32) * 1.0
        cls_weights = negative_cls_weights + 1.0 * positives.type(torch.float32)
        pos_normalizer = positives.sum(1, keepdim=True).type(torch.float32)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_losses = focal_loss_val._compute_loss(cls_preds, one_hot_targets, cls_weights.cuda())  # [N, M]

        cls_losses_reduced = cls_losses.sum()/example['labels'].shape[0]
        cls_losses_reduced = cls_losses_reduced.detach().cpu().numpy()

    else:
        cls_losses_reduced = 1000
```
