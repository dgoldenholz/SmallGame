# SmallGame
a small test game


To run:

# Create the environment named 'vision_env' with Python 3.10
conda create -n vision_env python=3.10 -y

# Activate the environment
conda activate vision_env

pip install "pygame>=2.5.0" "opencv-python>=4.8.0" "mediapipe>=0.10.0" "numpy>=1.24.0"

python3 small_game.py

the first time you run it from terminal, you will be asked to give permission to camera for terminal. say yes.
if you don't do that, you can always use the keyboard for controls.

