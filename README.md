
# Gun Detector — Computer Vision Project

A computer vision application built with **Python, YOLO, OpenCV, and Streamlit**. The application allows users to upload images or videos and view model-generated detection results.

## Features

- Image upload and detection
- Video upload and frame-by-frame processing
- Annotated image and video previews
- Detection confidence settings
- Downloadable processed results
- CSV export of detection information
- Interactive web interface powered by Streamlit

## Technologies Used

- Python
- Ultralytics YOLO
- OpenCV
- Streamlit
- Pillow
- NumPy
- Pandas

## Project Structure

```text
Client/
├── app.py
├── requirements.txt
└── best.pt
```

- `app.py` — Main Streamlit application.
- `requirements.txt` — Python dependencies.
- `best.pt` — Trained model weights. Keep this file locally or provide it through a secure model-storage solution if it is too large for GitHub.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/malik-tanveer/Gun_detector_assigment
cd Gun_detector_assigment/Client
```

Replace the example values with your actual GitHub repository URL and folder name.

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Model

Place your trained `best.pt` file in the same directory as `app.py`, unless your code uses a different model path.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Requirements

Your `requirements.txt` should contain:

```text
ultralytics
streamlit
opencv-python-headless
pillow
numpy
pandas
```

## Deployment

This project can be deployed using Streamlit Community Cloud.

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app and select your repository.
4. Set the app file path to `Client/app.py` if `Client` is a subfolder.
5. Ensure the required dependencies and model weights are accessible during deployment.
6. Deploy the application and check the logs if any errors occur.

**Note:** Large model files may exceed GitHub's file-size limits. Use an appropriate model-storage solution when needed.

## Important Disclaimer

This project is an educational computer vision demonstration. Its predictions may be incorrect or incomplete and should not be treated as proof that a weapon is present or absent. It is not a substitute for professional security procedures or human verification.

## Future Improvements

- Improve the user interface and error handling.
- Add model performance metrics.
- Display detection confidence and processing statistics.
- Improve deployment reliability.
- Add support for additional object-detection datasets.

## Author

**Malik Tanveer**

GitHub: [malik-tanveer](https://github.com/malik-tanveer)

---

*Built as a computer vision and machine learning learning project.*
