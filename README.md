<div align="center">

# 💊 MediVision AI

### Real-Time Medicine Package Recognition Using Computer Vision

**Developed by Milaf Ali ALjuaid**

</div>

---

## 📌 Project Idea

MediVision AI is a Computer Vision project developed using Python and OpenCV.

The system uses a camera to recognize registered medicine and supplement packages in real time.

The project is designed to identify only the packages stored in the system. If an unknown object or an unregistered package is shown to the camera, the system displays:

> **NO MEDICINE PACKAGE DETECTED**

---

## 🎯 Project Goal

The main goal of this project is to demonstrate how Computer Vision can be used for real-time package recognition.

The system compares the package shown in front of the camera with previously registered reference images.

---

## ⚙️ How the System Works

The system follows these steps:

1. Open the computer camera.
2. Capture the live video.
3. Detect visual features from the camera image.
4. Compare the detected features with the registered reference images.
5. Use feature matching to determine the closest registered package.
6. Verify the detected package using geometric matching.
7. Confirm the detection across several consecutive frames.
8. Display the package information if it is successfully recognized.
9. Display **NO MEDICINE PACKAGE DETECTED** if no registered package is detected.

---

## 💊 Registered Products

| Product | Category | Main Content |
|---|---|---|
| Redoxon Vitamin C 1000 mg | Vitamin / Dietary Supplement | Vitamin C |
| NOW Omega-3 Fish Oil | Dietary Supplement | Omega-3 (EPA / DHA) |
| Prof 400 mg | Pain Relief Medicine | Ibuprofen 400 mg |
| Panadol 500 mg | Pain Relief Medicine | Paracetamol |

> **Educational disclaimer:** This project is for Computer Vision and educational purposes only. It does not provide medical advice, dosage recommendations, or treatment instructions.

---

## 🧠 Computer Vision Method

The project uses **ORB (Oriented FAST and Rotated BRIEF)** to detect distinctive visual features from the registered package images and the live camera frames.

The system then uses:

- ORB Feature Detection
- BFMatcher
- Hamming Distance
- KNN Feature Matching
- Lowe's Ratio Test
- Homography
- RANSAC
- Inlier Verification
- Temporal Stability

These techniques help reduce false detections and prevent random objects from being incorrectly recognized as registered packages.

---

## 🔍 Recognition Logic

The system does not depend only on the number of matching points.

Instead, it performs several verification steps:

```text
Camera Frame
     ↓
ORB Feature Detection
     ↓
Feature Matching
     ↓
Lowe Ratio Test
     ↓
Homography Calculation
     ↓
RANSAC Verification
     ↓
Inlier Verification
     ↓
Frame Stability Check
     ↓
Package Recognition
```
## ✨ Main Features

📷 Real-time camera recognition
💊 Recognition of registered packages
🔎 ORB feature detection
🧠 Computer Vision matching
🛡️ False detection reduction
📊 Verified feature count
⏱️ Frame stability verification
🚫 Unknown object rejection
🖥️ Live information display
🐍 Python implementation
👁️ OpenCV processing

---

## 🛠️ Technologies Used

| Technology | Purpose                               |
| ---------- | ------------------------------------- |
| Python     | Main programming language             |
| OpenCV     | Computer Vision and camera processing |
| NumPy      | Numerical processing                  |
| ORB        | Feature detection and description     |
| BFMatcher  | Feature matching                      |
| RANSAC     | Geometric verification                |
| Webcam     | Real-time image input                 |

---

## 📁 Project Structure

```text
MedicineVision
│
├── medicine_vision.py
│
└── references
    ├── redoxon.jpg
    ├── omega3.jpg
    ├── prof.jpg
    └── panadol.jpg
```

---

## 💻 Installation

Make sure Python is installed on your computer.

Then install the required libraries:
```text
pip install opencv-python numpy
```

---

## code:
```text
import cv2
import os

# ==========================================
# MEDICINE VISION SYSTEM
# ==========================================

products = {
    "redoxon": {
        "name": "Redoxon Vitamin C 1000 mg",
        "category": "Vitamin / Dietary Supplement",
        "content": "Vitamin C"
    },

    "omega3": {
        "name": "NOW Omega-3 Fish Oil",
        "category": "Dietary Supplement",
        "content": "Omega-3 (EPA / DHA)"
    },

    "prof": {
        "name": "Prof 400 mg",
        "category": "Pain Relief Medicine",
        "content": "Ibuprofen 400 mg"
    },

    "panadol": {
        "name": "Panadol 500 mg",
        "category": "Pain Relief Medicine",
        "content": "Paracetamol"
    }
}


# ==========================================
# LOAD REFERENCE IMAGES
# ==========================================

reference_folder = "references"

orb = cv2.ORB_create(
    nfeatures=2500,
    scaleFactor=1.2,
    nlevels=8
)

reference_data = {}

print("\nLoading reference images...\n")

for product_id in products:

    image_path = os.path.join(
        reference_folder,
        product_id + ".jpg"
    )

    image = cv2.imread(image_path)

    if image is None:
        print("ERROR: Could not load:", image_path)
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    keypoints, descriptors = orb.detectAndCompute(
        gray,
        None
    )

    if descriptors is not None:

        reference_data[product_id] = {
            "image": image,
            "keypoints": keypoints,
            "descriptors": descriptors
        }

        print(
            f"{product_id} loaded - "
            f"{len(keypoints)} features"
        )


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open camera.")
    exit()


# ==========================================
# MATCHER
# ==========================================

matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING
)


# ==========================================
# STABILITY SYSTEM
# ==========================================

last_product = None
stable_frames = 0

REQUIRED_STABLE_FRAMES = 5


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.resize(
        frame,
        (1000, 700)
    )

    gray_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    keypoints_frame, descriptors_frame = (
        orb.detectAndCompute(
            gray_frame,
            None
        )
    )


    # ======================================
    # DEFAULT RESULT
    # ======================================

    best_product = None
    best_inliers = 0
    best_good_matches = 0


    # ======================================
    # COMPARE CAMERA WITH PRODUCTS
    # ======================================

    if descriptors_frame is not None:

        for product_id, data in reference_data.items():

            reference_descriptors = data["descriptors"]
            reference_keypoints = data["keypoints"]

            try:

                # Find two nearest matches
                matches = matcher.knnMatch(
                    reference_descriptors,
                    descriptors_frame,
                    k=2
                )

                # Lowe's ratio test
                good_matches = []

                for pair in matches:

                    if len(pair) == 2:

                        m, n = pair

                        if m.distance < 0.70 * n.distance:

                            good_matches.append(m)


                # Need enough matches
                if len(good_matches) < 12:
                    continue


                # ==================================
                # HOMOGRAPHY / GEOMETRIC VERIFICATION
                # ==================================

                src_points = []
                dst_points = []

                for match in good_matches:

                    src_points.append(
                        reference_keypoints[
                            match.queryIdx
                        ].pt
                    )

                    dst_points.append(
                        keypoints_frame[
                            match.trainIdx
                        ].pt
                    )


                if len(src_points) >= 8:

                    src_points = (
                        __import__("numpy")
                        .float32(src_points)
                        .reshape(-1, 1, 2)
                    )

                    dst_points = (
                        __import__("numpy")
                        .float32(dst_points)
                        .reshape(-1, 1, 2)
                    )


                    H, mask = cv2.findHomography(
                        src_points,
                        dst_points,
                        cv2.RANSAC,
                        5.0
                    )


                    if mask is not None:

                        inliers = int(
                            mask.sum()
                        )

                        # Only accept strong geometric matches
                        if (
                            inliers >= 10
                            and
                            inliers > len(good_matches) * 0.45
                        ):

                            if inliers > best_inliers:

                                best_inliers = inliers

                                best_good_matches = (
                                    len(good_matches)
                                )

                                best_product = product_id


            except Exception:
                pass


    # ======================================
    # TEMPORAL STABILITY
    # ======================================

    current_product = best_product


    if current_product == last_product:

        if current_product is not None:
            stable_frames += 1
        else:
            stable_frames = 0

    else:

        stable_frames = 0

        last_product = current_product


    # ======================================
    # FINAL DETECTION
    # ======================================

    detected = False

    if (
        best_product is not None
        and
        stable_frames >= REQUIRED_STABLE_FRAMES
    ):

        detected = True


    # ======================================
    # USER INTERFACE
    # ======================================

    # Background panel
    cv2.rectangle(
        frame,
        (15, 15),
        (985, 205),
        (25, 25, 25),
        -1
    )


    # Title
    cv2.putText(
        frame,
        "MEDICINE VISION",
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2
    )


    # ======================================
    # NO MEDICINE
    # ======================================

    if not detected:

        cv2.putText(
            frame,
            "NO MEDICINE PACKAGE DETECTED",
            (35, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Show a registered package to the camera",
            (35, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1
        )

        cv2.putText(
            frame,
            "Registered products: 4",
            (35, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1
        )


    # ======================================
    # MEDICINE DETECTED
    # ======================================

    else:

        product = products[best_product]

        cv2.putText(
            frame,
            "MEDICINE PACKAGE DETECTED",
            (35, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Product: " + product["name"],
            (35, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Type: " + product["category"],
            (35, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "Content: " + product["content"],
            (500, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Verified Features: {best_inliers}",
            (500, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 210, 210),
            1
        )


    # ======================================
    # CAMERA WINDOW
    # ======================================

    cv2.imshow(
        "Medicine Vision System",
        frame
    )


    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLOSE
# ==========================================

camera.release()
cv2.destroyAllWindows()

print("\nMedicine Vision stopped.")
```

---

## ▶️ How to Run the Project

Open the project folder in VS Code or Command Prompt.

Run:
```text
python medicine_vision.py
```
The camera window will open automatically.

Show one of the registered packages to the camera.

---

## 📊 Detection Verification

The system uses a minimum number of feature matches and geometric verification.

A package is not immediately accepted after one matching frame.

Instead, the same package must remain detected across multiple consecutive frames.

This improves recognition stability and reduces accidental detections.

---

## 🚫 Unknown Package Handling

One of the main features of MediVision AI is rejecting objects that are not included in the reference database.

For example:

Registered Package → Recognized ✅

Unknown Package → Not Detected ❌

Book → Not Detected ❌

Phone → Not Detected ❌

Random Object → Not Detected ❌

---


## 🔮 Future Improvements

Possible future improvements include:

Adding more registered products
Using a trained deep-learning model
Adding OCR for package text
Improving recognition under different lighting conditions
Supporting different camera angles
Adding a graphical user interface
Saving recognition results
Adding a package detection bounding box
Creating a mobile application
Improving recognition speed

---

## 📚 Project Type

Computer Vision Project

Field

Computer Engineering

Main Areas
- Artificial Intelligence
- Computer Vision
- Image Processing
- Python Programming
- OpenCV


## ⚠️ Disclaimer

This project is created for educational and Computer Vision demonstration purposes.

The displayed product information is identification metadata only and should not be used as medical advice or as a basis for taking any medicine or supplement.
