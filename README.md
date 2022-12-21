## PCB Defect Detection using YOLOv5
Our journal paper, "A Deep Context Learning based PCB Defect Detection Model with Anomalous Trend Alarming System" has been accepted by the journal, Results in Engineering (RINENG) and can be found [here](https://www.google.com). 

## Sample Inference Results
<img src="/samples/missinghole_1.jpg" width="200"> <img src="/samples/missinghole_2.jpg" width="200"> <img src="/samples/missinghole_3.jpg" width="200"> <img src="/samples/missinghole_4.jpg" width="200"> <img src="/samples/mousebite_1.jpg" width="200"> <img src="/samples/mousebite_2.jpg" width="200"> <img src="/samples/mousebite_3.jpg" width="200"> <img src="/samples/mousebite_4.jpg" width="200"> <img src="/samples/opencircuit_1.jpg" width="200"> <img src="/samples/opencircuit_2.jpg" width="200"> <img src="/samples/opencircuit_3.jpg" width="200"> <img src="/samples/opencircuit_4.jpg" width="200"> <img src="/samples/short_1.jpg" width="200"> <img src="/samples/short_2.jpg" width="200"> <img src="/samples/short_3.jpg" width="200"> <img src="/samples/short_4.jpg" width="200"> <img src="/samples/spur_1.jpg" width="200"> <img src="/samples/spur_2.jpg" width="200"> <img src="/samples/spur_3.jpg" width="200"> <img src="/samples/spur_4.jpg" width="200"> <img src="/samples/spurious_1.jpg" width="200"> <img src="/samples/spurious_2.jpg" width="200"> <img src="/samples/spurious_3.jpg" width="200"> <img src="/samples/spurious_4.jpg" width="200">

### Details of the dataset:
1) The dataset contains 10,668 naked PCB images, containing 6 common defects: missing hole, mouse bite, open circuit, short circuit, spur and spurious copper.
2) All images are scaled from 600x600 to 608x608 for training and testing purposes.

## Getting Started
Check out this [tutorial](https://github.com/ultralytics/yolov5/blob/develop/tutorial.ipynb) from the official YOLOv5 GitHub page for more information.
- Clone this repository
~~~
git clone https://github.com/JiaLim98/YOLO-PCB.git
~~~

- Prepare Python environment using [Anaconda3](https://www.anaconda.com/download/).
- Install all dependencies in [requirements.txt](https://github.com/JiaLim98/YOLO-PCB/blob/main/requirements.txt).
~~~
conda install pytorch==1.7.1 torchvision -c pytorch
pip install opencv-python
~~~

## Dataset Preparation
All dataset must be put alongside the `yolov5` folder. For YOLO dataset format, please organize them as follows:
~~~
yolov5/
PCB_dataset/
  - images/
    - trainPCB/
      - *.jpg
    - valPCB/
      - *.jpg
    - testPCB/
      - *.jpg
  - labels/
    - trainPCB/
      - *.txt
    - valPCB/
      - *.txt
    - testPCB/
      - *.txt
~~~

## Training and Testing
- Training with YOLOv5 pretrained weights. Note: RTX3070 holds a maximum batch size of 34 with 8GB VRAM
~~~
python train.py --img 608 --batch 34 --epochs 3 --data PCB.yaml --weights yolov5s.pt --nosave --cache
~~~
- Testing with finalized weights. Note: YOLOv5 uses batch size of 32 by default
~~~
python test.py --weights ./weights/baseline_fpn_loss.pt --data PCB.yaml --img 608 --task test --batch 1
~~~

## Inference
1) Place desired image in the `data` folder with any subfolder name. For example:
~~~
data/
  - PCB/
    - *.jpg
~~~
2) Run model inference with finalized weights on desired data folder.
~~~
python test.py --source ./data/PCB/ --weights ./weights/baseline_fpn_loss.pt --conf 0.9
~~~

## Anomalous Trend Alarming System
The featured anomalous trend detection algorithms can be found [here](https://github.com/JiaLim98/YOLO-PCB/blob/main/utils/alarm.py). To pair it with your own model, simply import the functions during model inference or detection.
- [`size`](https://github.com/JiaLim98/YOLO-PCB/blob/main/utils/alarm.py#L5-L60) detects for increasing detection sizes (i.e. defects are increasingly bigger)
- [`rep`](https://github.com/JiaLim98/YOLO-PCB/blob/main/utils/alarm.py#L62-L127) detects for repetitive local defect occurrences (i.e. defects are experiencing buildup at a similar point continuosly)
- [`cnt`](https://github.com/JiaLim98/YOLO-PCB/blob/main/utils/alarm.py#L129-L165) detects for increasing defect occurrences (i.e. similar defects repeatedly occurs, regardless of its location)

## Acknowledgement
All credits of YOLOv5 go to the [Ultralytics](https://github.com/ultralytics) company. Official YOLOv5 GitHub repository can be found [here](https://github.com/ultralytics/yolov5).

All credits of the dataset go to RunWei Ding, LinHui Dai, GuangPeng Li and Hong Liu, authors of "TDD-net: a tiny defect detection network for printed circuit boards". Their repository can be found [here](https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB).
