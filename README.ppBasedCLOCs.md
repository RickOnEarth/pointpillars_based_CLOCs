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


### 运行：
```python
python ./pytorch/train_Re.py train --config_path=./configs/pointpillars/car/CLOCs_xyres_16.proto --model_dir=model_dirs/model_dir_CLOCs --pickle_result=True
python ./pytorch/train_Re.py evaluate --config_path=./configs/pointpillars/car/CLOCs_xyres_16.proto --model_dir=model_dirs/model_dir_CLOCs --pickle_result=True/False
```
