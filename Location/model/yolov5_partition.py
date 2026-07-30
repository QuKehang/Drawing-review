import cv2
import numpy as np
import os
import json


class Colors:
    def __init__(self):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


class yolov5:
    def __init__(self, onnx_path, confThreshold=0.25, nmsThreshold=0.45):
        self.classes = ['figue', 'annotation', 'title', 'title bar', 'draw']
        self.colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(self.classes))]
        num_classes = len(self.classes)
        self.anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.nl = len(self.anchors)
        self.na = len(self.anchors[0]) // 2
        self.no = num_classes + 5
        self.stride = np.array([8., 16., 32.])
        self.inpWidth = 640
        self.inpHeight = 640
        self.net = cv2.dnn.readNetFromONNX(onnx_path)

        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, ratio, (dw, dh)

    def box_area(self, boxes):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def box_iou(self, box1, box2):
        area1 = self.box_area(box1)
        area2 = self.box_area(box2)
        lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
        rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
        wh = rb - lt
        wh = np.maximum(0, wh)
        inter = wh[:, :, 0] * wh[:, :, 1]
        iou = inter / (area1[:, np.newaxis] + area2 - inter)
        return iou

    def numpy_nms(self, boxes, scores, iou_threshold):
        idxs = scores.argsort()
        keep = []
        while idxs.size > 0:
            max_score_index = idxs[-1]
            max_score_box = boxes[max_score_index][None, :]
            keep.append(max_score_index)

            if idxs.size == 1:
                break
            idxs = idxs[:-1]
            other_boxes = boxes[idxs]
            ious = self.box_iou(max_score_box, other_boxes)
            idxs = idxs[ious[0] <= iou_threshold]

        keep = np.array(keep)
        return keep

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def non_max_suppression(self, prediction, conf_thres=0.25, agnostic=False):
        xc = prediction[..., 4] > conf_thres
        min_wh, max_wh = 2, 4096
        max_nms = 30000
        output = [np.zeros((0, 6))] * prediction.shape[0]

        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue
            x[:, 5:] *= x[:, 4:5]
            box = self.xywh2xyxy(x[:, :4])
            conf = np.max(x[:, 5:], axis=1)
            j = np.argmax(x[:, 5:], axis=1)
            re = np.array(conf.reshape(-1) > conf_thres)
            conf = conf.reshape(-1, 1)
            j = j.reshape(-1, 1)
            x = np.concatenate((box, conf, j), axis=1)[re]
            n = x.shape[0]
            if not n:
                continue
            elif n > max_nms:
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]
            c = x[:, 5:6] * (0 if agnostic else max_wh)
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = self.numpy_nms(boxes, scores, self.nmsThreshold)
            output[xi] = x[i]
        return output

    def detect(self, srcimg, img_index):
        im = srcimg.copy()
        im, ratio, wh = self.letterbox(srcimg, self.inpWidth, stride=self.stride, auto=False)
        blob = cv2.dnn.blobFromImage(im, 1 / 255.0, swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]
        pred = self.non_max_suppression(outs, self.confThreshold, agnostic=False)

        figue_boxes = []

        for i in pred[0]:
            left = int((i[0] - wh[0]) / ratio[0])
            top = int((i[1] - wh[1]) / ratio[1])
            width = int((i[2] - wh[0]) / ratio[0])
            height = int((i[3] - wh[1]) / ratio[1])
            conf = i[4]
            classId = int(i[5])
            label = '%s: %.2f' % (self.classes[classId], conf)

            if self.classes[classId] in self.classes:  # Check if the class is in the custom list
                figue_boxes.append({
                    'label': label,
                    'coordinates': {
                        'left': left,
                        'top': top,
                        'right': width,
                        'bottom': height
                    }
                })

            cv2.rectangle(srcimg, (left, top), (width, height), colors(classId, True), 5, lineType=cv2.LINE_AA)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv2.putText(srcimg, label, (left - 20, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)

        self.figures_boxes = figue_boxes  # Store the figue_boxes info in the instance

        for box in figue_boxes:
            print(f"Image {img_index} - Label: {box['label']}")
            print(f"Image {img_index} - Coordinates: left={box['coordinates']['left']}, top={box['coordinates']['top']}, right={box['coordinates']['right']}, bottom={box['coordinates']['bottom']}")

        return srcimg


def mult_test(onnx_path, img_dir, img_save_root_path, json_save_root_path, video=False):
    model = yolov5(onnx_path)
    
    # 确保保存图像和 JSON 文件的目录存在
    if not os.path.exists(img_save_root_path):
        os.mkdir(img_save_root_path)
    
    if not os.path.exists(json_save_root_path):
        os.mkdir(json_save_root_path)
    
    # 遍历图像目录
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            # 生成图像文件的路径
            image_path = os.path.join(root, file)
            
            # 生成保存图像和 JSON 文件的路径
            save_img_path = os.path.join(img_save_root_path, file)
            save_json_path = os.path.join(json_save_root_path, f'{os.path.splitext(file)[0]}.json')
            
            # 读取图像
            srcimg = cv2.imread(image_path)
            # 对图像进行检测
            annotated_img = model.detect(srcimg, file)
            
            # 保存带标注的图像
            cv2.imwrite(save_img_path, annotated_img)

            # 保存检测结果到 JSON 文件
            figue_boxes = model.figures_boxes  # 假设你有办法访问这些信息
            with open(save_json_path, 'w') as f:
                json.dump(figue_boxes, f, indent=4)

            print("Finished processing:", file)

