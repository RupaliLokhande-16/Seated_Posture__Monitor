
import cv2
import mediapipe as mp
import json
import argparse
import time
import math
from collections import deque
from statistics import median

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "pose_landmarker_lite.task"

# Posture detection settings
CALIBRATION_SECONDS = 5
TORSO_ANGLE_LIMIT = 10.0
HEAD_OFFSET_LIMIT = 0.18
SMOOTHING_WINDOW = 15
MIN_VOTES = 5
GOOD_VOTE_RATIO = 0.60


def extract_features(landmarks):
    """
    Extract scale-normalized posture measurements
    from a side-view seated person.
    """

    nose = landmarks[0]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]
    right_hip = landmarks[24]

    points = [
        nose,
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip
    ]

    for point in points:
        visibility = getattr(point, "visibility", 1.0)

        if visibility is None or visibility < 0.5:
            return None

    # Calculate shoulder center
    shoulder_x = (
        left_shoulder.x + right_shoulder.x
    ) / 2

    shoulder_y = (
        left_shoulder.y + right_shoulder.y
    ) / 2

    # Calculate hip center
    hip_x = (
        left_hip.x + right_hip.x
    ) / 2

    hip_y = (
        left_hip.y + right_hip.y
    ) / 2

    torso_height = abs(hip_y - shoulder_y)

    if torso_height < 0.05:
        return None

    # Torso angle relative to vertical
    horizontal_difference = abs(shoulder_x - hip_x)

    torso_angle = math.degrees(
        math.atan2(horizontal_difference, torso_height)
    )

    # Head position normalized by torso height
    head_offset = abs(nose.x - shoulder_x) / torso_height

    return torso_angle, head_offset


def classify_posture(features, baseline):
    """
    Compare current posture with calibrated
    upright posture.
    """

    torso_angle, head_offset = features

    baseline_angle, baseline_head = baseline

    angle_difference = torso_angle - baseline_angle

    head_difference = head_offset - baseline_head

    if (
        angle_difference > TORSO_ANGLE_LIMIT
        or head_difference > HEAD_OFFSET_LIMIT
    ):
        return "bad"

    return "good"


def monitor_posture(
    source,
    duration,
    output_file,
    preview=False,
    calibration_seconds=CALIBRATION_SECONDS
):

    camera = cv2.VideoCapture(source)

    if not camera.isOpened():
        print("Error: Could not open video source.")
        return 1

    is_video_file = not isinstance(source, int)

    fps = camera.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30.0

    posture_history = deque(maxlen=SMOOTHING_WINDOW)

    calibration_samples = []

    good_seconds = 0.0
    bad_seconds = 0.0

    posture_breaks = 0
    previous_posture = None

    total_frames = 0
    valid_frames = 0

    start_time = time.monotonic()
    previous_time = start_time

    frame_index = 0
    baseline = None

    options = PoseOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    try:

        with PoseLandmarker.create_from_options(options) as landmarker:

            while True:

                if not is_video_file:

                    if time.monotonic() - start_time >= duration:
                        break

                success, frame = camera.read()

                if not success:
                    break

                if frame is None or frame.size == 0:
                    continue

                total_frames += 1

                # Calculate frame time
                if is_video_file:

                    current_time = frame_index / fps
                    elapsed = 1.0 / fps

                else:

                    now = time.monotonic()

                    elapsed = min(
                        now - previous_time,
                        0.2
                    )

                    current_time = now - start_time
                    previous_time = now

                frame_index += 1

                try:

                    rgb_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame
                    )

                    timestamp_ms = int(current_time * 1000)

                    result = landmarker.detect_for_video(
                        image,
                        timestamp_ms
                    )

                    label = "NO PERSON DETECTED"
                    color = (0, 255, 255)

                    if result.pose_landmarks:

                        landmarks = result.pose_landmarks[0]

                        features = extract_features(landmarks)

                        if features is not None:

                            valid_frames += 1

                            # First five seconds: upright calibration
                            if current_time < calibration_seconds:

                                calibration_samples.append(features)

                                label = "CALIBRATING - SIT UPRIGHT"
                                color = (0, 255, 255)

                            else:

                                # Calculate baseline after calibration
                                if baseline is None:

                                    if not calibration_samples:

                                        print(
                                            "Error: No valid calibration "
                                            "observations found."
                                        )

                                        return 1

                                    baseline = (
                                        median(
                                            sample[0]
                                            for sample in calibration_samples
                                        ),
                                        median(
                                            sample[1]
                                            for sample in calibration_samples
                                        )
                                    )

                                    print(
                                        "\nCalibration completed."
                                    )

                                    print(
                                        "Baseline torso angle:",
                                        round(baseline[0], 2)
                                    )

                                    print(
                                        "Baseline head offset:",
                                        round(baseline[1], 3)
                                    )

                                prediction = classify_posture(
                                    features,
                                    baseline
                                )

                                posture_history.append(prediction)

                                # Majority voting for stability
                                stable_posture = None

                                if len(posture_history) >= MIN_VOTES:

                                    good_votes = posture_history.count("good")
                                    bad_votes = posture_history.count("bad")

                                    total_votes = len(posture_history)

                                    if (
                                        good_votes / total_votes
                                        >= GOOD_VOTE_RATIO
                                    ):

                                        stable_posture = "good"

                                    elif (
                                        bad_votes / total_votes
                                        >= GOOD_VOTE_RATIO
                                    ):

                                        stable_posture = "bad"

                                # Accumulate observed time
                                if stable_posture == "good":

                                    good_seconds += elapsed

                                    label = "GOOD POSTURE"
                                    color = (0, 255, 0)

                                elif stable_posture == "bad":

                                    bad_seconds += elapsed

                                    label = "BAD POSTURE"
                                    color = (0, 0, 255)

                                else:

                                    label = "CHECKING POSTURE"
                                    color = (0, 255, 255)

                                # Count good-to-bad transitions
                                if (
                                    stable_posture == "bad"
                                    and previous_posture == "good"
                                ):

                                    posture_breaks += 1

                                if stable_posture is not None:

                                    previous_posture = stable_posture

                    # Display preview
                    if preview:

                        cv2.putText(
                            frame,
                            label,
                            (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            color,
                            2
                        )

                        cv2.imshow(
                            "Seated Posture Monitor",
                            frame
                        )

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                except Exception as error:

                    print("Skipping frame:", error)
                    continue

    except Exception as error:

        print("Posture detection error:", error)
        return 1

    finally:

        camera.release()

        if preview:
            cv2.destroyAllWindows()

    # Validate observations
    total_observed_seconds = good_seconds + bad_seconds

    if (
        valid_frames == 0
        or baseline is None
        or total_observed_seconds <= 0
    ):

        print("No usable posture observations found.")
        return 1

    # Calculate final score
    posture_score = (
        good_seconds / total_observed_seconds
    ) * 100

    bad_percentage = (
        bad_seconds / total_observed_seconds
    ) * 100

    if is_video_file:
        session_duration = frame_index / fps
    else:
        session_duration = time.monotonic() - start_time

    # Final JSON report
    report = {

        "session_duration_seconds": round(
            session_duration, 2
        ),

        "total_frames": total_frames,

        "valid_frames": valid_frames,

        "calibration_seconds": calibration_seconds,

        "good_posture_seconds": round(
            good_seconds, 2
        ),

        "bad_posture_seconds": round(
            bad_seconds, 2
        ),

        "good_posture_percentage": round(
            posture_score, 2
        ),

        "bad_posture_percentage": round(
            bad_percentage, 2
        ),

        "posture_breaks": posture_breaks,

        "posture_score": round(
            posture_score, 2
        )
    }

    with open(output_file, "w") as file:

        json.dump(report, file, indent=4)

    print("\nSession completed!")

    print(json.dumps(report, indent=4))

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Seated Posture Monitor"
    )

    parser.add_argument(
        "--source",
        default="0"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=60
    )

    parser.add_argument(
        "--out",
        default="posture.json"
    )

    parser.add_argument(
        "--preview",
        action="store_true"
    )

    parser.add_argument(
        "--calibration",
        type=int,
        default=5
    )

    args = parser.parse_args()

    video_source = (
        int(args.source)
        if args.source.isdigit()
        else args.source
    )

    exit_code = monitor_posture(
        video_source,
        args.duration,
        args.out,
        args.preview,
        args.calibration
    )

    raise SystemExit(exit_code)
