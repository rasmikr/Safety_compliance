# 🦺 PPE Detection — Construction Safety Compliance

Detect Personal Protective Equipment (PPE) on construction sites using **YOLO** and the [Construction-PPE dataset](https://docs.ultralytics.com/datasets/detect/construction-ppe/).

The model identifies **11 classes** — both worn and missing PPE — enabling real-time safety compliance monitoring.

| Worn PPE (✅ Compliant) | Missing PPE (❌ Violation) |
|---|---|
| helmet | no_helmet |
| gloves | no_gloves |
| vest | no_boots |
| boots | no_goggle |
| goggles | none |
| Person | — |

---

## 📁 Project Structure

```
safety_compliance/
├── config/
│   ├── __init__.py
│   ├── dataset.yaml        # Dataset paths & class definitions
│   └── settings.py         # Central project settings
├── utils/
│   ├── __init__.py
│   └── visualization.py    # Annotation & compliance helpers
├── train.py                # Train YOLO on Construction-PPE
├── predict.py              # Run inference (image/video/webcam)
├── evaluate.py             # Evaluate model metrics
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

Optional env controls (copy `.env.example` to `.env`):

- `DISPLAY_CLASSES=helmet,boots` shows only selected classes in detection overlays.
- `SHOW_INFERRED_VIOLATIONS=false` disables strict `no_*` overlays.
- `FILTER_INFERRED_BY_DISPLAY_CLASSES=true` filters strict `no_*` labels to match `DISPLAY_CLASSES`.

### Code defaults vs `state/config.yaml` (why “two defaults” is misleading)

Sometimes documentation reads as if there are **two competing defaults**: paths fixed in Python (e.g. `PROJECT_ROOT / "files" / "input"` and `PROJECT_ROOT / "files" / "output"`) **and** separate “defaults” in `state/config.yaml` such as `default_input_path` / `default_output_path`.

That wording is easy to misread. There is **one conceptual default for developers**: the paths and tuning values defined **in code** (and merged defaults in [`storage/config_io.py`](storage/config_io.py)). The YAML entries `default_input_path` and `default_output_path` exist so a **deploy or the HTTP API** can **override** watch/output roots **without editing Python** (for example `/data/inbox` and `/data/out` under Docker). Phrases like “use YAML if set, else fall back to code” describe **optional runtime overrides on top of code defaults**, not a second, independent notion of “what the default is.”

In short: **code defaults are canonical**; persisted config is how operators **optionally replace** those values for a given environment.

> The dataset (~178 MB) will auto-download on the first training run.

---

## 🏋️ Training

```bash
# Default: 100 epochs, 640px, yolo11n.pt
python train.py

# Quick test run
python train.py --epochs 5 --imgsz 320 --batch 8

# Use a larger model
python train.py --model yolo11s.pt --epochs 100

# Resume interrupted training
python train.py --resume
```

Trained weights are saved to `runs/detect/ppe_detection/weights/best.pt`.

---

## 🔍 Inference

```bash
# Single image
python predict.py --source path/to/image.jpg --show

# Folder of images (save annotated results)
python predict.py --source path/to/images/ --save

# Video
python predict.py --source video.mp4 --save

# Webcam (live)
python predict.py --source 0 --show

# Custom model + confidence
python predict.py --source image.jpg --model runs/detect/ppe_detection/weights/best.pt --conf 0.4
```

---

## 📊 Evaluation

```bash
# Evaluate on validation set
python evaluate.py

# Evaluate on test set with per-class breakdown
python evaluate.py --split test --verbose

# Custom model
python evaluate.py --model path/to/best.pt --split test --verbose
```

Reports **mAP@0.5**, **mAP@0.5:0.95**, **Precision**, and **Recall**.

---

## 📚 References

- [Construction-PPE Dataset — Ultralytics Docs](https://docs.ultralytics.com/datasets/detect/construction-ppe/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Dataset Download](https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip)

---

## 📄 License

This project uses the Construction-PPE dataset provided under the [AGPL-3.0 License](https://ultralytics.com/license) by Ultralytics.


i need to following changes . Complete usecase
user should be able to register ( which means we should consider that folder while processing videos or images , and detect the ppe ) and user should be able to unregister a folder ( which means , u dont have to consider that folder for detection , we may skip that ) . User should also be able to send a single file location and op location and we should be able to process that . There should be a default input op folder configurable by user . 

API working 
Registering and unregistering can be done through an api , so it will craete a registery file if not exists , otherwise will add the  files to register ( from input ) to  that registery( csv may be ) . and in unregister api , it should be able to access same csv file and unregister the folder ( remove that folder entry from the registery or mar kit as unregistered ) . if the user wants to register any existing registery again , we may ask can we proceseed with same op folder or not , if same just change the tag , otherwise change the op location saved in registery as well . For registering default ip op folder we may use one api , and set it in env . so that folder will be added in registery . 


now the processing can be done in same scheduler.py ( lets reuse the function , but for now create another scheduler.py , or keep one as reference ) and do the same kind of processing , do the detection for all the new files in registered folders . 


now we need some payload changes in apis . 
1) while setting configuration for the first time ( default config )we may accept the following ips :  input , op locations when registering , also should be able to configure (frame_no,frame_skip_rate,video_fps) inrespective supported media type . if not given we should have some default values from programmer side 
2) detection should support video and images . ( also lets store supported media files eg: (in video mp4, mkv etc) and in image (img, jpeg, etc )unsupported media files in these categories . 
3) user should be able to pass what to detect , eg: boots , helmets , glouse , all etc 
4) user should be able to pass input , op locations when registering , also should be able to configure (frame_no,frame_skip_rate,video_fps) in respective supported media type . if not given it will continue with default 


data storing structure 
1) we may create seperate folders for image and video ( give the structure ) 
2) we may store folder registery and file registery in a particular location
3) default values in a dir and in yaml , decide it structure . 
4) scheudler refrence file creation 


so lets strat with a prooper plan and architecture . very minor changes will be required in the overall concept of scheduler . anyway give the plan . thie above is the entire usecase . after plan finalisation we can build each task , test it and move on to another  