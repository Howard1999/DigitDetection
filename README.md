<h2>Reproduce Submit</h2>
<hr>

<h3>1. Enviornment</h3>
<h4>python version:</h4>
3.7.12<br>
<h4>package:</h4>
pytorch=1.10.0<br>
torchvision=0.11.1<br>
h5py<br>
ptqdm<br>
seaborn<br>
opencv<br>
openmim<br>
mvcc-full<br>
https://github.com/open-mmlab/mmdetection <br>
https://github.com/ultralytics/yolov5 <br>

<h3>2. Model link</h3>
<a href="https://drive.google.com/file/d/1imEOPP2W7Q71aW2wBBwRIWvts5VkIrMJ/view?usp=sharing">
Download Link
</a>

<h3>3. Inference Example</h3>
<a href="https://drive.google.com/file/d/12ovdZO3WeczAX1Xlv2KsDkMP7nzVrH6r/view?usp=sharing">
Colab
</a>
<br>
or
<br>

``` python
from inference import inference
'''
assume folder structure
./
 |model.pt
 |data/
      |test/...  << test image folder
 |yolov5/...     << git cloned yolov5
'''
inference('./model.pt', './data/test', './answer.json') # model_path, test_path, output_path
```
