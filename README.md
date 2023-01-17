# NIA50 Docker 이미지 제출용 코드
---
# 폴더 구조

```
NIA50
  ├── Data
  │	  ├── 50-1
  │	  ├── 50-2
  │		   ├── images_2d
  │		   ├── images_2d_test
  │		   ├── ImageSets_2d
  │		   ├── labels_2d
  │
  ├── TSAI
  ├── OpenPCDet
  ├── YOLOv5
           ├── cfg
	       │    ├── nia50_data_yolov5l6.yaml
	       │	├── nia50_model_yolov5l6.yaml
	       │
           ├── ckpt  
           │	├── nia50_bestweights_yolov5l6.pt
           │
           ├── model(yolov5)
           ├── result
           ├── yolov5.py
```
