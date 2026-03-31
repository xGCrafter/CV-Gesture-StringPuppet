# CV-Guesture-StringPuppet
A Marionette String Puppet created in your webcam using openCV and Mediapipe using hand landmarking, which detects the tip of your 5 fingers to render a stick puppet on the screen, with a few more fun features.

# Demo
![Demo](./demo.gif)

# Requirements & Installation
- Python 3.9 or higher
- opencv-python, mediapipe and pygame

```
git clone https://github.com/xGCrafter/CV-Guesture-StringPuppet
cd CV-Guesture-StringPuppet
pip install -r requirements.txt
python3 main.py
```
# Finger Controls
Each fingertip corresponds to a specific "string" on the puppet, specifically:
1. Thumb Finger -> Left Knee
2. Index Finger -> Left Hand
3. Centre Finger -> Head
4. Ring Finger -> Right Hand
5. Pinky Finger -> Right Knee

# How To Use
After running main.py, bring your hand into the camera frame and you will see a stick puppet attached to your hand via a few thin strings, which can be used as a real marionette puppet.

# Special Reaction
If the Index and Ring fingers (ie the two hands) go above the middle finger (ie the head), a text appears displaying "PARTYYY!!" and a disco jam audio file begins playing. 
This is an example of how hand landmarking can be used :)


I spent 3 days on this project and if you enjoy it, leave a star 
