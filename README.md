# Seated Posture Monitor

## About the Project

I developed this project to check a person's sitting posture using Python and computer vision. It uses MediaPipe to detect body points like the head, shoulders and hips, and checks whether the person is sitting straight or leaning forward.

The program also tracks how long the person maintains good or bad posture and gives a final posture score.

## Tools Used

- Python
- OpenCV
- MediaPipe
- NumPy

## How It Works

The program takes input from a webcam or recorded video. For the first 5 seconds, the person needs to sit straight so the system can learn their normal posture.

After that, it checks changes in body position and classifies the posture as good or bad. At the end, it saves the results in a JSON file.

## How to Run

First, install the required libraries:

```bash
pip install -r requirements.txt
```

Download the MediaPipe Pose Landmarker Lite model and save it as `pose_landmarker_lite.task` in the project folder.

To run with a video:

```bash
python posture_monitor.py --source posture_test4.mp4 --duration 60 --out posture.json --preview
```


## Output

The `posture.json` file contains the session duration, good and bad posture time, number of posture breaks and an overall score out of 100.

## Note

The system works best with a side-view video where the upper body is clearly visible. It is a basic posture-monitoring project, not a medical tool.
