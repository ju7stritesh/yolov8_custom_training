from ultralytics import YOLO
import cv2
import os


dir_path = 'Videos/'
idx_frame = 0

labels = ['Amazon-logo', 'Amazon-truck', 'Amazon-van', 'Amazon-word', 'Fedex-logo', 'Fedex-truck', 'Fedex-van', 'Fedex-word', 'Target-logo', 'Target-truck', 'Target-word', 'UPS-Van', 'UPS-logo', 'UPS-truck', 'UPS-van', 'USPS-logo', 'USPS-truck', 'USPS-van', 'logo', 'truck', 'van']


def predict_on_image(model, img, conf):
    result = model(img, conf=conf)[0]

    # detection
    cls = result.boxes.cls.cpu().numpy()    # cls, (N, 1)
    probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
    boxes = result.boxes.xyxy.cpu().numpy()   # box with xyxy format, (N, 4)

    return boxes, cls, probs


model = YOLO("best_logo.pt")

for file in os.listdir(dir_path):
    print ("test", file)
    file_path = dir_path + file
    final_result = {}
    final_result['boxes'] = []
    final_result['labels'] = []
    final_result['prob'] = []
    cam = cv2.VideoCapture(file_path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    im_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        idx_frame += 1
        ret, img1 =  cam.read()
        if ret == False:
            break
        boxes, cls, probs = predict_on_image(model, img1, conf=0.20)
        print (boxes, probs)
        for t in range(len(boxes)):
           [x1,y1,x2,y2] = boxes[t]
           label = labels[int(cls[t])]
           # if label == 'person' or label == 'car' or label == 'truck' or label == 'bus' or label == 'potholes':
           if label in labels and 'UPS' in label:
              conf = int(probs[t]*100)
              final_result['boxes'].append([x1,y1,x2,y2])
              final_result['labels'].append(label)
              final_result['prob'].append(conf)
              cv2.rectangle(img1, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 1)  # filled
              cv2.putText(img1, label + ' - ' + str(conf), (int(x1),int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        # half = cv2.resize(img1, (0, 0), fx = 0.5, fy = 0.5)
        print (final_result)
        cv2.imshow("test", img1)
        # if not os.path.exists(outdir):
        #    os.makedirs(outdir)
        # cv2.imwrite(outdir + file, img1)
        cv2.waitKey(0)

