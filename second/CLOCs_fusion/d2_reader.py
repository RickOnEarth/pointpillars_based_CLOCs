import pathlib
import numpy as np

def detection_2d_reader(img_idx):
    detection_2d_result_path = pathlib.Path("../d2_detection_data_yolo_nms_0.6")
    img_idx = "{:06d}".format(img_idx)
    detection_2d_file_name = f"{detection_2d_result_path}/{img_idx}.txt"
    with open(detection_2d_file_name, 'r') as f:
        lines = f.readlines()
    content = [line.strip().split(' ') for line in lines]
    predicted_class = np.array([x[0] for x in content], dtype='object')
    predicted_class_index = np.where(predicted_class=='Car')
    detection_result = np.array([[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    score = np.array([float(x[15]) for x in content])  # 1000 is the score scale!!!
    f_detection_result=np.append(detection_result,score.reshape(-1,1),1)
    middle_predictions = f_detection_result[predicted_class_index, :].reshape(-1, 5)
    top_predictions = middle_predictions[np.where(middle_predictions[:, 4] >= -100)]

    return top_predictions