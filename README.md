# Face Mask Detection (Simple Haar Cascade Demo)

A small, beginner-friendly Python + OpenCV project. It opens your webcam,
detects faces in real time using a **Haar Cascade**, and then makes a **simple
guess** about whether each person is wearing a face mask.

> ⚠️ **disclaimer**
> A Haar Cascade only detects **faces**. It has **no idea** what a mask is.
> The mask / no-mask decision in this project is a **basic image-processing
> heuristic** (it looks at skin colour), **not** a trained machine-learning
> model. It is intended as an **educational demonstration** and will make
> mistakes depending on lighting, skin tone, facial hair, glasses, camera
> angle, mask colour, and so on. It is **not** a production-grade mask detector.

## 1. What the project does

- Opens your default webcam and shows the live video.
- Detects every face in each frame (supports multiple people at once).
- For each face, guesses **mask** or **no mask** and draws:
  - a **green** box + `MASK DETECTED`, or
  - a **red** box + `NO MASK` and a `PLEASE WEAR A MASK` warning.
- Quits cleanly when you press **`q`**.

## 2. How the face detection works

OpenCV ships with pre-trained **Haar Cascade** classifiers. This project uses
`haarcascade_frontalface_default.xml`. Each webcam frame is:

1. Converted to grayscale (Haar Cascades work on grayscale images).
2. Passed to `detectMultiScale()`, which scans the image at multiple sizes and
   returns a rectangle `(x, y, w, h)` for every face it finds.

We then draw a box around each returned rectangle.

## 3. How the simple mask heuristic works

This is the intentionally simple, non-ML part:

1. Split each detected face box into an **upper half** (eyes / forehead) and a
   **lower half** (nose / mouth / chin).
2. Convert each half to the **YCrCb** colour space and count how many pixels
   fall inside a typical **skin-colour** range.
3. The upper half is almost always bare skin. If the **lower half** shows
   **much less skin** than the upper half, we assume something (a mask) is
   covering it and report **MASK DETECTED**. Otherwise we report **NO MASK**.

The sensitivity is controlled by `MASK_SKIN_RATIO_THRESHOLD` near the top of
`main.py`. Lower it to make "mask" harder to trigger; raise it to make it
easier.

Because it relies purely on skin colour, it can be fooled by (for example) a
hand over the mouth, a dark beard, unusual lighting, or a skin-coloured mask.
That is expected for a heuristic like this.

## 4. Installing the dependencies

You need Python 3 installed. Then, from this folder:

```bash
pip install -r requirements.txt
```

This installs `opencv-python` (which also provides NumPy).

## 5. Running the program

From inside this project folder:

```bash
python main.py
```

A window titled **"Face Mask Detection (Haar Cascade demo)"** should appear
showing your webcam feed with boxes and labels.

### Where is the Haar Cascade file?

You do **not** need to download it manually. OpenCV bundles the cascade inside
the installed `opencv-python` package, and the program loads it automatically
from `cv2.data.haarcascades`.

If you prefer to ship the file with the project, download
`haarcascade_frontalface_default.xml` (from the official OpenCV GitHub repo,
`opencv/data/haarcascades/`) and place it **right next to `main.py`**:
```

`main.py` checks for a local copy next to itself **first**, and only falls back
to the bundled OpenCV copy if it doesn't find one.

## 6. How to exit

Click the video window to focus it, then press the **`q`** key. The program
stops the loop, releases the webcam, and closes all windows.

## 7. Limitations

- **Not a real mask detector.** The mask decision is a skin-colour heuristic,
  not a trained model, so accuracy is limited.
- **Frontal faces only.** The cascade detects faces looking roughly at the
  camera; profiles and tilted heads are often missed.
- **Sensitive to lighting and skin tone.** The skin-colour range is fixed, so
  very bright, very dark, or coloured lighting reduces accuracy.
- **Fooled by look-alikes.** A hand over the mouth, a heavy beard, or a
  skin-coloured mask can all confuse the heuristic.
- **One camera, one machine.** It uses the default local webcam (`CAMERA_INDEX = 0`).

OpenCV module is not very accurate , for a more accurate program we need a machine learning model with multiple datasets to work properly.
