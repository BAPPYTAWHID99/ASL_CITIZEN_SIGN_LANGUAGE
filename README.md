# ASL Citizen Sign Language Recognition

A Django web application for recognizing isolated ASL signs from 200-frame,
450-feature landmark sequences. The deployed inference model is a BiGRU with
attention; a Transformer checkpoint is included for comparison.

## Dataset

The processed ASL Citizen keypoint dataset remains hosted on Hugging Face:
[ASL Citizen Processed Keypoints-200](https://huggingface.co/datasets/SharoonArshad/asl-citizen-processed-200).
The dataset files are not included in this repository. The app's trained
checkpoints and label map are included so inference does not require the full
dataset.

## Run locally

Python 3.10 or later is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-django.txt
python manage.py runserver
```

Open <http://127.0.0.1:8000/>. The prediction endpoint accepts a JSON
`sequence` with shape `(200, 450)`.

## Training

Download the processed dataset from Hugging Face and extract it into this
project directory so its `shards/` folder is alongside the training scripts.
Then run `python train_bigru.py` or `python train_transformer.py`.
