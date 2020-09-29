import threading
import cv2
import datetime
import numpy as np
import time, re
from urllib.request import urlopen
from PIL import ImageFont, ImageDraw, Image
import imutils
import requests

class ESC32CAM:
    def __init__(self, url, fix_size, rotate=0, wait_restart=30):
        self.shutdown = False
        self.url_ready = True
        self.fix_size = fix_size
        self.url = url
        self.rotate = rotate
        self.success_stream = time.time()
        self.wait_restart = wait_restart

        blank = np.zeros((fix_size[1], fix_size[0], 3), dtype = 'uint8')
        self.blank = blank.copy()
        self.blank_readerr = self.printText(blank.copy(), url+'讀取錯誤', color=(0,255,255,0), size=0.35, pos=(100,100), type="Chinese")
        self.blank_restart = self.printText(blank.copy(), url+'重啟中', color=(0,255,255,0), size=0.35, pos=(100,100), type="Chinese")

        self.image = blank.copy()

    def chek_url(self, url):
        try:
            c = requests.get(url, timeout=6)
            return True
        except:
            return False      

    def printText(self, bg, txt, color=(0,255,0,0), size=0.7, pos=(0,0), type="Chinese"):
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


    def restart_stream(self):
        print("restart ", self.url)
        self.success_stream = time.time()
        self.image = self.blank_restart.copy()
        self.quit()
        self.run()

    def read_stream(self):
        
        if(self.url_ready is False):

            if(time.time()-self.success_stream>self.wait_restart):
                self.restart_stream()
        
        fix_size = self.fix_size
        rotate = self.rotate

        try:
        #if self.chek_url(self.url):
            stream = urlopen(self.url, timeout=6)
            self.success_stream = time.time()
            #ret, img = stream.read()
            imgNp=np.array(bytearray(stream.read()),dtype=np.uint8)
            img=cv2.imdecode(imgNp,-1)
            self.url_ready = True
            
        #else:
        #except Exception as e:
        except:
            #print(e, "error-->", self.url)
            print("read ",self.url,"stream error...")
            self.url_ready = False
            #height,width = 0, 0
            img = self.blank_readerr.copy()


        height,width = img.shape[:2]
        
        ip = re.findall( r'[0-9]+(?:\.[0-9]+){3}', self.url )
        img_txt = self.printText(img.copy(), ip[0], color=(255,0,0,0), size=0.65, pos=(10,25), type="Chinese")
        now_date = datetime.date.today().strftime("%B %d%Y, %B %d %I:%M%p")
        img_txt = self.printText(img_txt, now_date, color=(0,255,0,0), size=0.55, pos=(width-340, height-45), type="Chinese")

        self.org_img_size = (width, height)
        img = cv2.resize(img, (fix_size[0], fix_size[1]))
        img_txt = cv2.resize(img_txt, (fix_size[0], fix_size[1]))
        if(rotate>0):
            img = imutils.rotate(img, rotate)
            img_txt = imutils.rotate(img_txt, rotate)

        self.image = img

        return True


    def looprun(self):
        while not self.shutdown:
            self.read_stream()

        self.quit()

    def run(self):
        #with concurrent.futures.ThreadPoolExecutor() as executor:
        #    executor.submit(self.read_stream())
        self.threads=threading.Thread(target = self.looprun)
        self.threads.start()
    
    def quit(self):
        self.shutdown = True
        try:
            self.threads.join()
        except:
            pass
