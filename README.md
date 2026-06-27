# Face Recognition Attendance System

An automated attendance tracker that uses a webcam and face recognition to identify registered people and log their check-in time to a CSV file.

## Features

- Real-time face detection and recognition from a live webcam feed
- Roster-based enrollment using photos stored in the `Photos/` directory
- Automatic attendance logging with timestamps in `Attendance.csv`
- Duplicate prevention so each person is logged once per session
- On-screen visual feedback with bounding boxes and name labels

## Tech Stack

- **Python 3**
- **OpenCV** (`cv2`) — webcam capture, image processing, and display
- **face_recognition** — face detection and 128-dimensional face encodings (dlib-based)
- **NumPy** — distance calculations for best-match selection

## Project Structure

```
.
├── Project.py          # Main application
├── Attendance.csv      # Attendance log (NAME, TIME)
├── Photos/             # Registered face images (filename = person name)
│   ├── SRK.jpg
│   ├── Elon musk.jpg
│   ├── Virat Kohli.jpg
│   └── Bill Gates.jpg
└── README.md
```

## Prerequisites

- Python 3.8+
- A working webcam
- System dependencies for `face_recognition` / dlib (varies by OS)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/madhavs-1/face-detector.git
cd face-detector
```

2. Install Python dependencies:

```bash
pip install opencv-python face_recognition numpy
```

On some systems, `face_recognition` requires additional build tools or prebuilt dlib wheels. See the [face_recognition installation guide](https://github.com/ageitgey/face_recognition#installation) if you run into issues.

## Usage

1. Add one photo per person to the `Photos/` folder. Use the person's name as the filename (e.g. `John Doe.jpg`).

2. Run the application:

```bash
python Project.py
```

3. A webcam window titled **Cam** will open. When a registered face is detected, the system will:
   - Draw a bounding box around the face
   - Display the person's name
   - Append their name and current time to `Attendance.csv` (if not already logged)

4. Press `q` to quit.

## How It Works

1. **Enrollment** — At startup, all images in `Photos/` are loaded and converted to face encodings.
2. **Capture** — The webcam feed is read frame by frame.
3. **Optimization** — Each frame is downscaled to 25% size before inference to improve real-time performance.
4. **Detection** — Faces are located and encoded in the downscaled frame.
5. **Matching** — Encodings are compared against the roster using `compare_faces` and the closest match is selected via minimum face distance.
6. **Logging** — On a successful match, the name is written to `Attendance.csv` with a timestamp (`HH:MM:SS`), unless that name is already present in the file.

## Attendance Output

`Attendance.csv` uses this format:

```csv
NAME, TIME

SRK,19:09:34
ELON MUSK,19:09:55
```

## Google XYZ Formula (Project Impact)

When describing this project on a résumé or portfolio, use Google's XYZ formula:

> **Accomplished [X] as measured by [Y], by doing [Z].**

| Component | Meaning | Example for this project |
|-----------|---------|--------------------------|
| **X** | Outcome / impact | Automated real-time attendance logging |
| **Y** | Measurable proof | 4 registered users logged with unique timestamps and zero duplicate entries |
| **Z** | Method / actions | Integrated OpenCV webcam capture, dlib face encodings, and CSV persistence with deduplication |

**Sample bullet:**

> Built an automated face-recognition attendance system as measured by successful identification and timestamp logging of 4 registered individuals, by integrating OpenCV webcam capture, dlib-based face encodings via the face_recognition library, and CSV persistence with duplicate-entry prevention.

## Notes

- Ensure each enrollment photo contains a clear, front-facing face.
- Attendance deduplication is session-based (per run); delete or rotate `Attendance.csv` to start a fresh log.
- Performance depends on hardware; frame downscaling is used to keep inference responsive on consumer machines.

## License

This project is provided as-is for educational and personal use.
