# Final Project: Face Swapping
## Getting Started
### Dependency
 - opencv (3.4.2)
 - numpy (1.15.2)
 - scipy (1.1.0)
 - dlib (18.18)
### Usage

To evaluate, run `demo.py`. You can change `SOURCE_VIDEO_PATH` and `TARGET_VIDEO_PATH` to test on different videos.

### Note

The Face++ API requires `API_KEY` and `API_SECRET` to function, which are not included in this submission; The dlib face detector requires `shape_predictor_68_face_landmarks.dat` which are not included in this submission, either. Instead we saved the pre-calculated landmark detection results to `*.landmarks` and `*.facepp` files to speed up running. Use functions implemented in `loader.py` to save and load the pre-calculated data.