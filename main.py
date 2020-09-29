import threading, os
from libMulti import ESC32CAM
#from libMulti_urlopen import ESC32CAM
import cv2, time
import datetime
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from yoloOpencv import opencvYOLO
import imagezmq
import socket


#Camera setting
cam_resolution=(800,600)
#wait_restart_time: if last successful stream is over 'wait_restart_time' seconds, then restart the camera
wait_restart_time = 10

#Save img to the folder
collect_img = False
interval_seconds = 5
collect_img_path = "img_collected/"

#resize before send to ZMQ
resize_send = True
resize_zmq = (800,600)
'''
cam_data = [ ('http://172.30.17.192:81/stream', cam_resolution, buffer, 0, wait_restart_time),
             ('http://172.30.17.193:81/stream', cam_resolution, buffer, 0, wait_restart_time),
             ('http://172.30.17.194:81/stream', cam_resolution, buffer, 0, wait_restart_time),
             ('http://172.30.17.195:81/stream', cam_resolution, buffer, 0, wait_restart_time)
           ]
'''

cam_data = [ ('http://172.30.17.192/cam-hi.jpg', cam_resolution, 0, wait_restart_time),
             ('http://172.30.17.193/cam-hi.jpg', cam_resolution, 0, wait_restart_time),
             ('http://172.30.17.194/cam-hi.jpg', cam_resolution, 0, wait_restart_time),
             ('http://172.30.17.195/cam-hi.jpg', cam_resolution, 0, wait_restart_time)
           ]

enable_ai = False

#video record
write_video_output = True
record_time_period = [8, 19]
video_framerate = 24
video_split_interval = 60 * 60  #seconds
output_video_path = "meetingroom_videos/"
output_video_size = (1600, 1200)
labels_tw = { "person_head": "head", "person_vbox": "body" }

#------------------------------------------------------------------------
def printText(bg, txt, color=(0,255,0,0), size=0.7, pos=(0,0), type="Chinese"):
    (b,g,r,a) = color

    if(type=="English"):
        cv2.putText(bg,  txt, pos, cv2.FONT_HERSHEY_SIMPLEX, size,  (b,g,r), 2, cv2.LINE_AA)

    else:
        ## Use simsum.ttf to write Chinese.
        fontpath = "fonts/wt009.ttf"
        font = ImageFont.truetype(fontpath, int(size*10*4))
        img_pil = Image.fromarray(bg)
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos,  txt, font = font, fill = (b, g, r, a))
        bg = np.array(img_pil)

    return bg

def predict(img):
    print("Predict")
    yolo.getObject(img, labelWant="", drawBox=True, bold=2, textsize=0.95, bcolor=(255,255,255), tcolor=(0,255,255))
    width, height = img.shape[1], img.shape[0]

    for id, label in enumerate(yolo.labelNames):
        x = yolo.bbox[id][0]
        y = yolo.bbox[id][1]
        w = yolo.bbox[id][2]
        h = yolo.bbox[id][3]
        cx = int(x)
        if(cx>width): cx=width-60
        cy = int(y-h/3)
        if(cy<0): cy=0
        if(label=="bad"):
            txt_color = (0,0,255,0)
        elif(label=="none"):
            txt_color = (255,255,0,0)
        else:
            txt_color = (0,255,0,0)

        txt_size = round(w / 250, 1)
        #print(labels_tw[label], (w,h))
        img = printText(bg=img, txt=labels_tw[label], color=txt_color, size=txt_size, pos=(cx,cy), type="Chinese")

    return img

def write_video(img):
    out.write(img)

def write_collect(img, file_path):
    cv2.imwrite(file_path, img)

def check_env():
    if(collect_img is True):
        if (not os.path.exists(collect_img_path)):
            os.makedirs(collect_img_path)

    if(write_video_output is True):
        if (not os.path.exists(output_video_path)):
            os.makedirs(output_video_path)
#------------------------------------------------------------------------
window_name = 'ESP32CAM'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
record_time = 0
video_file_id = 0
last_collect_time = time.time()
cam_num = len(cam_data)
streams = []
for cam in cam_data:
    streams.append(
        ESC32CAM(url=cam[0], \
            fix_size=cam[1], \
            rotate=cam[2], \
            wait_restart=cam[3] )
    )

blank = np.zeros((cam_resolution[1], cam_resolution[0], 3), dtype = 'uint8')
out = None

if(enable_ai is True):
    yolo = opencvYOLO(modeltype="tiny", \
        objnames="models/yolov3-tiny/obj.names", \
        weights="models/yolov3-tiny/crowd_human.weights",\
        cfg="models/yolov3-tiny/yolov3-tiny.cfg")

for stream in streams:
    stream.run()

if __name__ == "__main__":
    check_env()
    port = 5555
    jpeg_quality = 95
    sender = imagezmq.ImageSender("tcp://*:{}".format(port), REQ_REP=False)
    rpi_name = socket.gethostname()


    while True:
        '''
        for id, stream in enumerate(streams):
            print(stream.url, stream.shutdown)
            if stream.shutdown is True or streams[id] is None:
                if(stream.chek_url(cam_data[id][0])):
                    print(cam_data[id][0], "is down but url ok, restart it.")
                    del streams[id]
                    stream = \
                        ESC32CAM(url=cam_data[id][0], \
                            fix_size=cam_data[id][1], \
                            buffer=cam_data[id][2], \
                            rotate=cam_data[id][3] )
                    streams.insert(id, stream)
                    
                    stream.run()
                else:
                    print(cam_data[id][0], "is down and url not ready.")
                    

            #    print("restart camera #id", id)
            #    streams[id].restart_stream()
        '''
        if(cam_num==1):
            combine = streams[0].image
        elif(cam_num==2):
            combine = np.vstack((streams[0].image, streams[1].image))
        elif(cam_num==3):
            #print(streams[0].image.shape, streams[1].image.shape, streams[2].image.shape)
            combine = np.vstack( (np.hstack((streams[0].image, streams[1].image)), np.hstack((streams[2].image, blank.copy()))) )
        elif(cam_num==4):
            combine = np.vstack( (np.hstack((streams[0].image, streams[1].image)), np.hstack((streams[2].image, streams[3].image))) )

        if(enable_ai is True):
            combine = predict(combine)


        if(write_video_output is True):
            dt = datetime.datetime.now()
            if(video_file_id==0 or (time.time()-record_time>video_split_interval)):
                if(video_file_id>0):
                    out.release()

                video_file_id += 1
                record_time = time.time()
                filename = "{}_{}.avi".format('{:%Y%m%d%H%M%S}'.format(dt), video_file_id)
                video_file = os.path.join(output_video_path, filename)
                print("created video file:", video_file)
                out = cv2.VideoWriter( video_file, fourcc, video_framerate, output_video_size)
            else:
                if(dt.hour<=record_time_period[1] and dt.hour>=record_time_period[0]):
                    thread_record = threading.Thread(target=write_video, args=(cv2.resize(combine, output_video_size),))
                    thread_record.start()
                    thread_record.join()

        if(collect_img is True and time.time()-last_collect_time>interval_seconds):
            last_collect_time = time.time()
            file_path = os.path.join(collect_img_path, str(time.time())+'.jpg')
            thread_collect = threading.Thread(target=write_collect, args=(combine,file_path,))
            thread_collect.start()
            thread_collect.join()

        if(resize_send is True):
            combine = cv2.resize(combine, resize_zmq)

        cv2.imshow(window_name, combine)
        ret_code, jpg_buffer = cv2.imencode( \
                ".jpg", combine, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        sender.send_jpg(rpi_name, jpg_buffer)

        k = cv2.waitKey(1)
        if k & 0xFF==ord('q'):

            for stream in streams:
                stream.quit()

            cv2.destroyAllWindows()
            if(write_video_output is True):
                out.release()

            break