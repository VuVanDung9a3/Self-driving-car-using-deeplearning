import time
import cv2
import argparse
import numpy as np
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True,
                help='path to input video')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()
conf_threshold = 0.5  # Confidence threshold
nms_threshold = 0.4  # Non-maximum suppression threshold
scale = 0.00392

# load names of classes
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet(args.weights, args.config)


# Get the name of output layers
def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


# function to draw bounding box on the detected object with class name
def draw_prediction(class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(frame, (x, y), (x_plus_w, y_plus_h), color, 2)


    cv2.putText(frame, "{:.2f}% {}".format(confidence * 100, label), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


# Post-processing the network output
def post_process(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    class_ids = []
    confidences = []
    boxes = []

# Thực hiện xác định bằng HOG và SVM
    start = time.time()

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                x = center_x - width / 2
                y = center_y - height / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        width = box[2]
        height = box[3]
        draw_prediction(class_ids[i], confidences[i], round(x), round(y), round(x + width), round(y + height))


# setup video
outFile = "yolo_output.avi"
if not os.path.isfile(args.video):
    print("Input video file ", args.video, " doesn't exits")
    sys.exit(1)
cap = cv2.VideoCapture(args.video)
# cap.set(3, frameWidth)      # width of the frames in the video
# cap.set(4, frameHeight)     # height of the frames in the video
outFile = args.video[:-4] + '_yolo_output.avi'
# write video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
write_out = cv2.VideoWriter(outFile, fourcc, 30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while(True):
    ret, frame = cap.read()
    # stop the program if reched end of video
    if not ret:
        print("Can't receive frame (stream end?). Exiting")
        print("Output file is stored is ", outFile)
        break

    # create a 4D blob from a frame
    blob = cv2.dnn.blobFromImage(frame, scale, (506, 506), (0, 0, 0), True, crop=False)

    # sets the input to the network
    net.setInput(blob)

    # run inference through the network and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    post_process(frame, outs)

    # Write the frame with the detection boxes
    write_out.write(frame.astype(np.uint8))

    #cv2.imshow("Result", frame)

#cap.release()
#cv2.destroyAllWindows()
