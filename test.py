if __name__ == '__main__':
    import argparse
    import multiprocessing as mp
    import stream
    import darknet as darknet
    import time
    from threading import Thread
else:
    import my_darknet.darknet as darknet
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from collections import deque
import yolov7_detect.detect as v7_detect
import math
import queue
from time import sleep
import time
import cv2
import numpy as np

class yolo_Detect:
    def __init__(self, weights, flag_yolo_length,conf-thres,iou-thres,classes, debug_flag = False):
        self.method_name = "Detect_shrimp"
        self.debug_flag = debug_flag
        self.flag_yolo_length = flag_yolo_length
        self.image_q = deque(maxlen=10)
        self.preprocess_res = deque(maxlen=10)
        self.detection_res = deque(maxlen=10)
        self.postprocess_res = deque(maxlen=10)
    #--yolo class
#        with open(classes, 'r') as f:
#            self.classes = [line.strip() for line in f.readlines()]
    #--weight config
#        self.weights = weights
#        self.config = config_file
#        self.data_file = data_file
        self.weights= weights
    #------------
    #--yolo setup----
#        self.network, self.class_names, self.class_colors = darknet.load_network(
#                self.config,
#                self.data_file,
#                self.weights,
#                batch_size=1
#            )
#        self.darknet_width = darknet.network_width(self.network)
#        self.darknet_height = darknet.network_height(self.network)
#        self.model = attempt_load(weights, map_location=0) # load FP32 model
#        self.model.half()
        self.conf-thres = conf-thres
        self.iou-thres = iou-thres
        self.classes = classes
    #----------------   
    #----yolov4-----------------
    def convert2relative(self,bbox):
        """
        YOLO format use relative coordinates for annotation
        """
        x, y, w, h  = bbox
        _height     = 640
        _width      = 640
        return x/_width, y/_height, w/_width, h/_height


    def convert2original(self,image, bbox):
        x, y, w, h = self.convert2relative(bbox)

        image_h, image_w, __ = image.shape

        orig_x       = int(x * image_w)
        orig_y       = int(y * image_h)
        orig_width   = int(w * image_w)
        orig_height  = int(h * image_h)

        bbox_converted = (orig_x, orig_y, orig_width, orig_height)

        return bbox_converted


    def convert4cropping(self,image, bbox):
        x, y, w, h = self.convert2relative(bbox)

        image_h, image_w, __ = image.shape

        orig_left    = int((x - w / 2.) * image_w)
        orig_right   = int((x + w / 2.) * image_w)
        orig_top     = int((y - h / 2.) * image_h)
        orig_bottom  = int((y + h / 2.) * image_h)

        if (orig_left < 0): orig_left = 0
        if (orig_right > image_w - 1): orig_right = image_w - 1
        if (orig_top < 0): orig_top = 0
        if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

        bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

        return bbox_cropping

,
    def detect(self,darknet_image,frame):
        td = time.time()
        detections = v7_detect.detect(self.weight, darknet_image,conf-thres,iou-thres,classes)
        print(f"{self.method_name} H2D & D2H: {time.time()- td:.5f}s")
        detections_calculate = [0]*len(self.class_names) #record amount of object
        
        #darknet.print_detections(detections, args.ext_output)
        darknet.free_image(darknet_image)
        detections_adjusted = []
        if darknet_image is not None:
            for label, confidence, bbox in detections:
                bbox_adjusted = self.convert2original(frame, bbox)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
                
                print(f'{str(label)}: {bbox_adjusted[2]},{bbox_adjusted[3]}')
                detections_calculate[self.class_names.index(str(label))]+=1 #record amount of object
            darknet_image = darknet.draw_boxes(detections_adjusted, frame, self.class_colors, self.flag_yolo_length)

        
        return darknet_image, detections_calculate
    #---------yolov4------end---

    def do_detection(self,frame, drawing):
        input_frame = frame.copy()
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
        frame_resized = cv2.resize(input_frame, (self.darknet_width, self.darknet_height),interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(self.darknet_width, self.darknet_height, 1)

        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        #image = self.detect(img_for_detect,frame)
        image,detections_calculate = self.detect(img_for_detect, frame.copy())
        #s ="detection result: "
        #for idx,count in enumerate(detections_calculate): #record amount of object
        #    s+=f"{self.class_names[idx]} : {count},"
        #print(s)
        if(drawing):
            return image, detections_calculate

        return detections_calculate

    def detection_full_loop(self):
        print(f"{self.method_name}: Function preprocess start")
        while True:
            try:
                td = time.time()
                image, stream_id = self.image_q.pop() 
            except:
                print(f"{self.method_name}: wait... preprocess")
                time.sleep(1/60)
                continue

            img, detections_calculate = self.do_detection(image, True)
            self.detection_res.appendleft([stream_id, img, detections_calculate])  
            fps = 1/(time.time()-td)
            print(f"Stream id {stream_id} {self.method_name} time: {(time.time()-td):.3f}s, fps {round(fps, 1)}")         

    def detection_full(self):
        try:
            td = time.time()
            image, stream_id = self.image_q.pop()
            img, detections_calculate = self.do_detection(image, True)
            self.detection_res.appendleft([stream_id, img, detections_calculate])  
            print(f"Stream id {stream_id} {self.method_name} time: {(time.time()-td):.3f}s")
        except:
            print(f"{self.method_name}: wait... preprocess")
            time.sleep(1/60)

    def preprocess(self):
        func_name = "preprocess"
        print(f"{self.method_name}: Function {func_name} start")
        while True:
            td = time.time()
            try:
                image, stream_id = self.image_q.pop()
                input_frame = image.copy()
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
                frame_resized = cv2.resize(input_frame, (self.darknet_width, self.darknet_height),interpolation=cv2.INTER_LINEAR)
                img_for_detect = darknet.make_image(self.darknet_width, self.darknet_height, 1)

                darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())

                self.preprocess_res.appendleft([stream_id, img_for_detect, image])
                if(self.debug_flag):
                    print(f"{self.method_name} preprocess: {(time.time() - td):.3f}s")
            except Exception as e:
                if(self.debug_flag):
                    print(f"{self.method_name} - {func_name}: {e}")
                    #print(f"{self.method_name}: wait... preprocess")
                time.sleep(1/60)
                continue

    def detection_normal(self):
        func_name = "detection_normal"
        try:
            td = time.time()
            stream_id, img_for_detect, image = self.preprocess_res.pop()
            detections = darknet.detect_image(self.network, self.class_names, img_for_detect, thresh=0.7)

            self.detection_res.appendleft([stream_id, image, img_for_detect, detections])
            if(self.debug_flag):
                print(f"Stream id {stream_id} {self.method_name} time: {(time.time()-td):.3f}s")
        except Exception as e:
            if(self.debug_flag):
                print(f"{self.method_name} - {func_name}: {e}")
            time.sleep(1/60)

    def detection_loop(self):
        func_name = "detection_loop"
        while True:
            try:
                td = time.time()
                stream_id, img_for_detect, image = self.preprocess_res.pop()
                detections = darknet.detect_image(self.network, self.class_names, img_for_detect, thresh=0.7)

                self.detection_res.appendleft([stream_id, image, img_for_detect, detections])
                if(self.debug_flag):
                    print(f"Stream id {stream_id} {self.method_name} time: {(time.time()-td):.3f}s")
            except Exception as e:
                if(self.debug_flag):
                    print(f"{self.method_name} - {func_name}: {e}")
                time.sleep(1/60)
                continue

    def postprocess(self):
        func_name = "postprocess"

        print(f"{self.method_name}: {func_name}")
        while True:
            td = time.time()
            try:        
                stream_id, image, img_for_detect, detections = self.detection_res.pop()
                
                detections_calculate = [0]*len(self.class_names) #record amount of object
                yolo_length = []
                #darknet.print_detections(detections, args.ext_output)
                darknet.free_image(img_for_detect)
                detections_adjusted = []
                if img_for_detect is not None:
                    for label, confidence, bbox in detections:
                        bbox_adjusted = self.convert2original(image, bbox)
                        shrimp_length = 0
                        if(self.flag_yolo_length):
                            _, _, w, h = bbox_adjusted
                            w = (w*32.5/800)*0.71
                            h = (h*21.875/450)*0.71
                            shrimp_length = round(math.sqrt((w*w + h*h)), 3)

                        detections_adjusted.append((str(label), confidence, bbox_adjusted, shrimp_length))
                        yolo_length.append(shrimp_length)
                        detections_calculate[self.class_names.index(str(label))]+=1 #record amount of object
                    
                    img_for_detect = darknet.draw_boxes(detections_adjusted, image, self.class_colors, self.flag_yolo_length)
                    
                if(self.flag_yolo_length):
                    size = np.array(yolo_length)
                    if len(size)==0 :
                        mean,median = 0,0
                    else:
                        mean = np.mean(size)
                        median = np.median(size)
                    shrimp_length_avg = [mean, median]
                    self.postprocess_res.appendleft([stream_id, img_for_detect, detections_calculate, shrimp_length_avg])
                else:
                    self.postprocess_res.appendleft([stream_id, img_for_detect, detections_calculate])

                if(self.debug_flag):
                    print(f"{self.method_name} postprocess: {(time.time() - td):.3f}s")
            except Exception as e:
                if(self.debug_flag):
                    print(f"{self.method_name} - {func_name}: {e}")
                    #print(f"{self.method_name}: wait... postprocess")
                time.sleep(1/60)
                continue

def task(q_in,source,bs): # not finish
    dataset = stream.LoadStreams(source)
    bs.value = len(dataset)  # batch_size
    
    #print(id(dataset))
    while True:
        #print ("in streaming")
        t1 = time.time()
        #for path,  im0s, counter,frame_Valid, fps in dataset:
        q_in.put(dataset.get_Item())
       
        #stream.my_Sleep(t1,30)


def reader(data,para):# not finish
    darknet_Detect = yolo_Detect(para[0], para[1], para[2], para[3])
    
    while True :
        if int(bs.value) != 0:
            result = [None]*int(bs.value)
            shrimp_d_time = [0]*int(bs.value)
            break
        time.sleep(1)

    print((result))
    '''
    #img = [None]*bs
    threads = [None] *int(bs.value)
    for i in range(0,int(bs.value)):
        #img[i] = img_container(False,None)
        img = img_container(False,None)
        threads[i] = Thread(target=detect, args=(para,img,i), daemon=True) #shrimp streaming        
        threads[i].start()

    while True:
        t1 = time.time()
        if not data.empty():            
            path,  im0s, counter,frame_Valid, fps = data.get()
            print(f"reader: {counter} {frame_Valid} {fps}")
            for i, det in enumerate(im0s):
                detect_calc_obj = "" 
                p, im0, isFrameValid = path[i], im0s[i].copy(), frame_Valid[i] # read frame
                #img[i] = im0
                img.status , img.img  = frame_Valid[i] , im0s[i].copy()
    '''            
    while True:
        #print ("out frames")
        t1 = time.time()
        if not data.empty():
            
            path,  im0s, counter,frame_Valid, fps = data.get()

            print(f"reader: {counter} {frame_Valid} {fps}")
            for i, det in enumerate(im0s):
                detect_calc_obj = "" 
                p, im0, isFrameValid = path[i], im0s[i].copy(), frame_Valid[i] # read frame
                ts = time.time()
                result[i],detections_calculate = darknet_Detect.do_Detect(im0.copy()) # do yolo detection (darknet)
                shrimp_d_time[i] = round(time.time() - ts,3) # shrimp detect time for each stream
                for idx,item in enumerate(detections_calculate):
                    detect_calc_obj +=f'{str(darknet_Detect.classes[idx])} : ({item}),'
                print(f'camera {i} -- {detect_calc_obj}, detection time:({shrimp_d_time[i]})s')
            print(f"detection total time: {sum(shrimp_d_time):.2f}")
            #for i, det in enumerate(result):           
            #    p= path[i]
            #    det = cv2.resize(det, (720, 480))
            #    cv2.imshow(f"camera -- {i}"+str(p), det.copy())
            #    cv2.waitKey(1)  # 1 millisecond
        #stream.my_Sleep(t1,30)

def detect(para,img,stream_idx):
    darknet_Detect = yolo_Detect(para[0], para[1], para[2], para[3])
    while(True):
        t1 = time.time()
        #print(f"stream index : {stream_idx}")
        #print(f'do shrimp detect {para[stream_idx]}')
        if(img.status):
            result,detections_calculate = darknet_Detect.do_Detect(img.img)

            print(f'{stream_idx}:({1/(time.time() - t1):.3f} FPS)')
        stream.my_Sleep(t1,60)
class img_container:
    def __init__(self, img_status,img):
        self.status = img_status
        self.img = img
'''
if __name__ == '__main__':
    weights = "yolov4-tiny_taifer_last.weights"
    config_file = "yolov4-tiny_dead.cfg"
    classes = "dead_shrimp.txt"
    data_file = "dead_shrimp.data"
    para =[classes,weights,config_file,data_file]
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='video_list.txt', help='file/dir/URL/glob, 0 for webcam')
    opt = parser.parse_args()

    source = str(opt.source)
    bs = mp.Value('d',0)
    #dataset = LoadStreams(source)
    #bs = len(dataset)  # batch_size
    #result[i],detections_calculate = darknet_Detect.do_Detect(im0.copy()) # do yolo detection (darknet)
    
    q = mp.Queue(maxsize=1)
    p1 = mp.Process(target=task, args=(q,source,bs), daemon = True)
    p2 = mp.Process(target=reader, args=(q,para), daemon = True)
    p1.start()
    #print(id(dataset))
    p2.start()
    while True:
    #    print("main hi")
        time.sleep(0.001)
'''

def upload_image(image, model):
    td = time.time() 
    
    model.do_detection(image, True)
    model.isDetect = True
    print(f"shrimp detect: {(time.time() - td):.5f}s ({1/(time.time() - td):.5f} FPS)")

def worker(work, model):
    release_counter = 0
    while release_counter < 100:
        if(not work.empty()):
            image = work.get()
            upload_image(image,model)
        else:
            release_counter += 1
            time.sleep(1/60)
    print("worker release")

def loader(work, image):
    work.put(image)

def my_Sleep(time_start,fps):
    time_end = time.time()
    if (((1/fps)-(time_end - time_start)) > 0): #let fps <=15 (0.066s)
        #print(f"sleep {(1/fps)-(time.time() - time_start):.3f}s")
        try:
            time.sleep((1/fps)-(time_end - time_start))
        except:
            print("wrong")

if __name__ == '__main__':
    weights = "best.pt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/home/eslab/Documents/Underwater_inference_system/shrimp_test.jpg",help="video source. If empty, uses webcam 0 stream")   
    parser.add_argument('--thread_count', default=1, type=int, help='for testing hardware stream amount')
    parser.add_argument('--delay', default=0, type=float, help='delay')
    parser.add_argument('--yolo_length', action='store_true', help='openpose shrimp length to yolo shrimp length')  
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    args = parser.parse_args()
    print("shrimp detection module")
    img = cv2.imread(args.input)

    shrimp_detect = yolo_Detect( weights, args.yolo_length,args.conf-thres,args.iou-thres,args.classes, debug_flag = False)
    Thread(target = shrimp_detect.preprocess).start() #preprocess image
    Thread(target = shrimp_detect.postprocess).start() # post process

    while True:
        shrimp_detect.image_q.appendleft([img, 0])

        shrimp_detect.detection_normal()
        try:
            current_id, tmp_shrimp_img, tmp_shrimp_data = shrimp_detect.postprocess_res.pop()
            print(f"shrimp: {current_id}")
        except:    
            print(f"detect_shrimp still detect...")
        sleep(1/15)

    engine_list =[]
    work_list =[]
    myWorker = []
    for i in range(0,args.thread_count):
        engine_list.append(yolo_Detect(classes, weights, config_file, data_file))
        work_list.append(queue.Queue())
    
    for i, s in enumerate(engine_list):
        upload_image(img, engine_list[i])
    print("warmup...")
    print("="*16,"sequential test","="*16)
    time.sleep(1)
    '''
    for j in range(0,10):
        td = time.time() 
        for i, s in enumerate(engine_list):
            upload_image(img, engine_list[i])

        t_end = (time.time() - td)
        print(f"{j+1} sequence whole detect: {t_end:.3f}s ({1/t_end:.3f} FPS)")
        print(f"sequence per detect: {t_end/len(engine_list):.3f}s ({1/(t_end/len(engine_list)):.3f} FPS)")
        print("="*50)
    print("="*16,"Concurent test","="*16)
    time.sleep(1)
    '''
    #j = 0
    while True:
        if(args.delay > 0):
            time.sleep(args.delay)
    #for j in range(0,10):
        td = time.time() 
        for i, s in enumerate(engine_list):
            upload_image(img, engine_list[i])

       # my_Sleep(td, 25)
        t_end = (time.time() - td)
        #print(f"{j+1} sequence whole detect: {t_end:.3f}s ({1/t_end:.3f} FPS)")
        print(f"sequence per detect: {t_end/len(engine_list):.5f}s ({1/(t_end/len(engine_list)):.5f} FPS)")
        print("="*50)
        #j +=1
    print("="*16,"Concurent test","="*16)
    time.sleep(1)

    #cv2.imshow('My Image', img)
    #while True:
    #    if cv2.waitKey() & 0xFF == ord('q'):
    #        print("Finish.")
    #        break

    #Doesn't work better
    '''
    for i, s in enumerate(engine_list):
        myWorker.append(Thread(target=worker, args=(work_list[i], engine_list[i])))
        myWorker[i].start()



    for j in range(0,10):
        for i, s in enumerate(engine_list):
            engine_list[i].isDetect = False    
        
        
        td = time.time() 
        for i, s in enumerate(engine_list):
            Thread(target=loader, args=(work_list[i], img), daemon = True).start()



        overDetect = 0

        while True:
            time.sleep(0.01)
            overDetect = 0
            for i, s in enumerate(engine_list):
                if(engine_list[i].isDetect == True):
                    overDetect +=1
                    
            if(overDetect == len(engine_list)):
                break
        
        t_end = (time.time() - td)
        print(f"{j+1} whole detect: {t_end:.3f}s ({1/t_end:.3f} FPS)")
        print(f"per detect: {t_end/len(engine_list):.3f}s ({1/(t_end/len(engine_list)):.3f} FPS)")
        print("="*50)
        '''
