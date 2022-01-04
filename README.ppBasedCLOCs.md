# pointpillars_based_CLOCs

## 2021-9-16
pointpillars based CLOCs,  v1.   
3D candidates cut from 107136 to 70400.   
NMS threshold 0.5, 0.5.   
Result slightly better then pointpillars only.


## 2021-10-15
pointpillars based CLOCs,  v2.   
3D candidates recover to 107136.
Fix problems with torch.from_numpy().type(), type_as(), LongTensor(). Transformation from different data types costs too much time.
Result slightly better then pointpillars only and v1.  


### run：
```python
python ./pytorch/train_Re.py train --config_path=./configs/pointpillars/car/CLOCs_xyres_16.proto --model_dir=model_dirs/model_dir_CLOCs_temp --pickle_result=True
python ./pytorch/train_Re.py evaluate --config_path=./configs/pointpillars/car/CLOCs_xyres_16.proto --model_dir=model_dirs/model_dir_CLOCs --pickle_result=True/False
```

## 2021-11-9
count the time   
nn: 10.9ms, processing before fusion: 21.5ms, fusion_layers: 2.4ms, post processing time: 5.7ms   
!!注意：evaluate时要将predict_kitti_to_anno()中的test_mode改为True!! loss的计算会占满cpu  
Training时test_mode置False


## 2021-11-15
fix the problem:  
all_3d_output_dict.update({'cls_preds': fusion_cls_preds_reshape})     
pretty good result:   
‘
Car AP@0.70, 0.70, 0.70:  
bbox AP:96.76, 96.31, 94.09  
bev  AP:93.85, 90.15, 89.80  
3d   AP:89.68, 80.70, 77.92  
aos  AP:96.56, 95.44, 92.93  
Car AP@0.70, 0.50, 0.50:  
bbox AP:96.76, 96.31, 94.09  
bev  AP:97.14, 96.81, 96.59  
3d   AP:97.12, 96.76, 96.46  
aos  AP:96.56, 95.44, 92.93  
’  
update train_stage2()/se.build_stage2_training_mx_v2()，high precision with fast speed（train_stage2()部分5.1ms）.  

## 2021-11-16  
在train_stage2()中，用anchors_mask与给定的score阈值来过滤box_preds，anchors_mask为False或score低于阈值的box_preds,  
不进行后面的decoding和IoU运算。这样精度不变，速度更快（train_stage2()部分4.3ms）。  
这个版本比较适合inference时用，training时用恐怕会有bug且training时不在意时间，感觉不要score_threshold更好。  
这版代码仅供后面部署的时候参考吧。文件命名为voxel.py.faster_inference  

## 2021-11-18  
d2_detection_data_*的路径添加到参数中   
train:
```bash
python ./pytorch/train_Re.py train --config_path=PATH --d2_path=../d2_detection_data_yolo_nms_0.6 --model_dir=PATH
``` 
注意将train_Re.py与voxelnet.py中的training_flag置True  

evaluate:
```bash
python ./pytorch/train_Re.py evaluate --config_path=PATH --d2_path=../d2_detection_data_yolo_nms_0.6 --model_dir=PATH
``` 
注意将train_Re.py与voxelnet.py中的training_flag置False  

inference:  
```bash
python ./pytorch/train_Re.py evaluate --config_path=PATH --d2_path=../d2_detection_data_yolo_nms_0.6 --model_dir=PATH --predict_test=True
```
注意将train_Re.py与voxelnet.py中的training_flag置False   
config中raw_data路径如下
```text
  kitti_info_path: "/mengxing/Data/Sets/raw_data/CLOCs_preprocess/object_format_2011_09_26_drive_0001/kitti_infos_test.pkl"
  kitti_root_path: "/mengxing/Data/Sets/raw_data/CLOCs_preprocess/object_format_2011_09_26_drive_0001"
```
