# Sentinel Vision — YOLO11 Streamlit App

1. Place your `best.pt` next to `app.py`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

The app shows weapon-related alerts only if the model's class names contain terms such as `gun`, `weapon`, `pistol`, `firearm`, or `rifle`. Edit `weapon_terms` in `app.py` if your dataset uses a different class name. The app does not make a definitive safety judgement.

Segmentation masks appear only when using a segmentation-trained YOLO model that returns `result.masks`. A detection-only model cannot produce real masks.
