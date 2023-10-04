import os
import json
import time
import csv
import cv2
import torch
import numpy as np
from pathlib import Path
import tensorflow as tf
import sys
sys.path.insert(0, '/Users/jihoon/venvs/yolov7-pose-tracking')
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_imshow, increment_path, set_logging, check_img_size, non_max_suppression, scale_coords
from utils.plots import draw_boxes
from utils.torch_utils import select_device, time_synchronized, TracedModel
from pose.utils.datasets import LoadImages as PoseLoadImages
from pose.detect import detect
from sort import Sort



def model_load(weights, device, imgsz):
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    return model, stride, imgsz


def view_image(p, im0):
    cv2.imshow(str(p), im0)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()


def img_prep(img, device, half):
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def track(det, img, im0, sort_tracker):
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    dets_to_sort = np.empty((0, 6))
    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
        dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))
    tracked_dets = sort_tracker.update(dets_to_sort)
    return tracked_dets


def predict_walk_type_with_confidence(model, skeleton_data):
    MAX_SEQUENCE_LENGTH = 632  # 학습 데이터의 최대 시퀀스 길이
    
    # Add this line to print the length and some values of skeleton_data
    # print("Length of skeleton_data:", len(skeleton_data))
    # print("First few values of skeleton_data:", skeleton_data[:10])
       
    # Convert the skeleton_data to a numpy array and reshape it to fit LSTM input shape
    skeleton_data = np.array(skeleton_data).reshape(1, 1, 34)
    
    # Adjust the padding to match training data length
    skeleton_data_padded = tf.keras.preprocessing.sequence.pad_sequences(skeleton_data, padding='post', dtype='float32', maxlen=MAX_SEQUENCE_LENGTH)
    skeleton_data_padded = skeleton_data_padded.reshape(1, MAX_SEQUENCE_LENGTH, 34)  # Assuming that the skeleton data has 34 features per timestep
    
    # 이제 패딩된 데이터를 사용하여 모델에서 예측
    prediction = model.predict(skeleton_data_padded)
    
    # 출력: skeleton_data_padded의 shape 및 데이터 타입
    # print("Skeleton data shape:", skeleton_data_padded.shape)
    # print("Skeleton data type:", skeleton_data_padded.dtype)
    
    #확률값 출력
    print("Prediction probabilities:", prediction[0])
    
    walk_type = np.argmax(prediction, axis=1)
    walk_labels = ["Parkinson's Walk", "Shuffling Walk", "Normal Walk"]
    return walk_labels[walk_type[0]]

def select_keypoints(kpts, remove_columns):
    # 모든 keypoints의 컬럼명 리스트를 여기에 적어주세요
    all_columns = ['nose-x', 'nose-y', "nose-conf",
            'left-eye-x', 'left-eye-y', "left-eye-conf",
            'right-eye-x', 'right-eye-y', "right-eye-conf",
            'left-ear-x', 'left-ear-y', "left-ear-conf",
            'right-ear-x', 'right-ear-y', "right-ear-conf",
            'left-shoulder-x', 'left-shoulder-y', "left-shoulder-conf",
            'right-shoulder-x', 'right-shoulder-y', "right-shoulder-conf",
            'left-elbow-x', 'left-elbow-y', "left-elbow-conf",
            'right-elbow-x', 'right-elbow-y', "right-elbow-conf",
            'left-hand-x', 'left-hand-y', "left-hand-conf",
            'right-hand-x', 'right-hand-y', "right-hand-conf",
            'left-hip-x', 'left-hip-y', "left-hip-conf",
            'right-hip-x', 'right-hip-y', "right-hip-conf",
            'left-knee-x', 'left-knee-y', "left-knee-conf",
            'right-knee-x', 'right-knee-y', "right-knee-conf",
            'left-foot-x', 'left-foot-y', "left-foot-conf",
            'right-foot-x', 'right-foot-y', "right-foot-conf"]
    
    indices_to_remove = [all_columns.index(col) for col in remove_columns]
    return [kpts[i] for i in range(len(kpts)) if i not in indices_to_remove]


def visualize(tracked_dets, im0, model, imgsz, stride, device, half, names, h5_model):
    bbox_xyxy = tracked_dets[:, :4]
    identities = tracked_dets[:, -1]
    categories = tracked_dets[:, 4]
    
    for idx, box in enumerate(bbox_xyxy):
        cat = int(categories[idx]) if categories is not None else 0
        if cat != 0:
            continue
        
        id = int(identities[idx]) if identities is not None else 0
        x1, y1, x2, y2 = [int(x) for x in box]
        obj = im0[y1:y2, x1:x2]
        if not obj.shape[0] or not obj.shape[1]:
            continue
        
        d = PoseLoadImages(obj, imgsz, stride)
        kpts, obj = detect(d, model, device, half, xy=[x1, y1])
        if kpts is None:
            continue
            
        # Print keypoints and skeleton_data for debugging
        # print("Keypoints for frame:", kpts)
        
        # 여기에서 kpts 값을 필터링
        remove_columns = ["nose-conf", "left-eye-conf", "right-eye-conf", "left-ear-conf", 
                          "right-ear-conf", "left-shoulder-conf", "right-shoulder-conf", 
                          "left-elbow-conf", "right-elbow-conf", "left-hand-conf", 
                          "right-hand-conf", "left-hip-conf", "right-hip-conf", 
                          "left-knee-conf", "right-knee-conf", "left-foot-conf", "right-foot-conf"]

        kpts = select_keypoints(kpts, remove_columns)
        
        kpts = np.array(kpts).reshape(1, -1)  # Assuming you are reshaping the data before feeding to model
        # print("Skeleton data for frame:", skeleton_data)

        # 이제 필터링된 kpts 값을 .h5 모델에 입력

        # Predict walk type based on keypoints
        walk_label = predict_walk_type_with_confidence(h5_model, kpts)
        
        # Draw the prediction result on the video frame
        label = f"ID: {id} Type: {walk_label}"
        im0 = cv2.putText(im0, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
        
    draw_boxes(im0, bbox_xyxy, identities, categories, names)
    return im0

def main(source):
    device = 'cpu'
    img_size, conf_thres, iou_thres = 640, 0.25, 0.45
    view_img, save_json = False, True
    save_img = not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # Initialize SORT tracker and save directory
    sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)
    save_dir = Path(increment_path(Path('output') / 'obj', exist_ok=False))
    save_dir.mkdir(parents=True, exist_ok=True)
    # Initialize device and model
    device = select_device(device)
    half = device.type != 'cpu'
    model, stride, imgsz = model_load('yolov7.pt', device, img_size)
    model_, stride_, imgsz_ = model_load('yolov7-w6-pose.pt', device, img_size)
    model = TracedModel(model, device, img_size)
    if half:
        model.half()
        model_.half()
    # Load the h5 model for walk prediction
    h5_model_path = "/Users/jihoon/venvs/skeleton-rnn/best_model_val_loss.h5"
    h5_model = tf.keras.models.load_model(h5_model_path)
    
    vid_path, vid_writer = None, None
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
        model_(torch.zeros(1, 3, imgsz_, imgsz_).to(device).type_as(next(model_.parameters())))
    t0 = time.time()
    nf = 0
    results = {}
    for path, img, im0s, vid_cap in dataset:
        nf += 1
        img = img_prep(img, device, half)
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]
        t2 = time_synchronized()
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=0)
        t3 = time_synchronized()
        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)
            save_path = str(save_dir / p.name)
            if len(det):
                tracked_dets = track(det, img, im0, sort_tracker)
                if len(tracked_dets) > 0:
                    results = visualize(tracked_dets, im0, model_, imgsz_, stride_, device, half, names, h5_model)
            else:
                tracked_dets = sort_tracker.update()
        print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        if view_img:
            view_image(p, im0)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
    if save_json:
        t4 = time.time()
        results['tag'] = {'time': t4-t0,
                          'num of frame': nf,
                          'total detected': len(results.keys()),
                          'frame/time': nf/(t4-t0)
                          }
        with open(save_path.split('.')[0]+'.json', 'w') as f:
            json.dump(results, f, indent=4)


def convert_json_to_csv(base='./output/'):
    path_list = [base + f for f in os.listdir(base) if os.path.isdir(base + f)]
    labels = ['nose-x', 'nose-y',
            'left-eye-x', 'left-eye-y',
            'right-eye-x', 'right-eye-y',
            'left-ear-x', 'left-ear-y',
            'right-ear-x', 'right-ear-y',
            'left-shoulder-x', 'left-shoulder-y',
            'right-shoulder-x', 'right-shoulder-y',
            'left-elbow-x', 'left-elbow-y',
            'right-elbow-x', 'right-elbow-y',
            'left-hand-x', 'left-hand-y',
            'right-hand-x', 'right-hand-y',
            'left-hip-x', 'left-hip-y',
            'right-hip-x', 'right-hip-y',
            'left-knee-x', 'left-knee-y',
            'right-knee-x', 'right-knee-y',
            'left-foot-x', 'left-foot-y',
            'right-foot-x', 'right-foot-y']  # Use the full labels list you provided before
    for path in path_list:
        files = [file for file in os.listdir(path) if file.endswith('.json')]
        for file in files:
            name = file.split('.')[0]
            with open(f'{path}/{file}', 'r') as j:
                data = json.load(j)
                frame = data['tag']['num of frame']
                for key in data.keys():
                    if len(data[key]) < (frame*0.5):
                        continue
                    with open(f'{path}/{name}-{key}.csv', 'w', newline='') as c:
                        w = csv.writer(c)
                        w.writerow(labels)
                        for kpts in data[key]:
                            w.writerow(kpts)


def process_video_and_visualize(source):
    with torch.no_grad():
        main(source)
    convert_json_to_csv()
    h5_model_path = "/Users/jihoon/venvs/skeleton-rnn/best_model_val_loss.h5"
    h5_model = tf.keras.models.load_model(h5_model_path)
    base = './output/'
    path_list = [base + f for f in os.listdir(base) if os.path.isdir(base + f)]
    for path in path_list:
        files = [file for file in os.listdir(path) if file.endswith('.csv')]
        for file in files:
            skeleton_data = np.loadtxt(f'{path}/{file}', delimiter=',', skiprows=1)
            predictions = predict_walk_type_with_confidence(h5_model, skeleton_data)
            # Add the prediction results on the video or save it separately.
            # ...


if __name__ == '__main__':
    source = '/Users/jihoon/venvs/skeleton-rnn/input'
    main(source)
    # process_video_and_visualize(video_source)