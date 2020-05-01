import argparse
from sys import platform
from utils.datasets import *
from utils.utils import *
from models import *  # set ONNX_EXPORT in models.py
from config import get_config
import glob
class Detector(object):
    def __init__(self):
        opt = get_config()
        self.img_size =opt.img_size
        weights= opt.weights
        self.device = torch_utils.select_device(opt.device)
        self.model = Darknet(opt.cfg, self.img_size)
        # Load weights 
        if weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(weights, map_location=self.device)['model'])
        else:  # darknet format
            load_darknet_weights(self.model,weights)

        self.model.to(self.device).eval()
        self.names = load_classes(opt.names)
        self.conf_thres=opt.conf_thres
        self.iou_thres=opt.iou_thres
        print(self.names)

    def detect(self,im0s):
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        t0 = time.time()
        # img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
        # _ = self.model(img.float()) if self.device.type != 'cpu' else img  # run once
        
        img = letterbox(im0s, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img,np.float32)  # uint8 to fp16/fp32
        img /= 255.0
        t = time.time()
        img = torch.from_numpy(img).cuda()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        #t1 = torch_utils.time_synchronized()
        pred = self.model(img)[0]
        #t2 = torch_utils.time_synchronized()
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        obj_types=[]
        box_detects=[]
        res=[]
        for i, det in enumerate(pred):  # detections per image
            im0 =im0s
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *x, conf, cls in det:
                    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                    obj_types.append(self.names[0])
                    # cv2.imwrite(str(uuid.uuid1())+".jpg",im0[c1[1]:c2[1],c1[0]:c2[0]])
                    top=c1[1]
                    left=c1[0]
                    right=c2[0]
                    bottom=c2[1]
                    box_detects.append(np.array([left, top, right, bottom]))
                    td=[c1[1],c2[1],c1[0],c2[0]]
                    height=c2[1]-c1[1]
                    res.append({'td':[int((int(x[0])+int(x[2]))/2),int((int(x[1])+int(x[3]))/2)],'td1':td,'height':height})
        print("Done !",time.time() - t0)
        return res,box_detects,obj_types


if __name__ == '__main__':
   
    detector=Detector()
    for path in glob.glob("test/*.*"):
        img=cv2.imread(path)
        res,boxes,types=detector.detect(img)
        print(len(boxes))
        font = cv2.FONT_HERSHEY_SIMPLEX 
        for box,typ in zip(boxes,types):
            img =cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),1,1)
            cv2.putText(img,typ,(box[0],box[1]),font,0.5,(255,0,0),1)
        cv2.imshow("image",img)
        cv2.waitKey(0)
