# NFL Health &amp; Safety - Helmet Assignment: Segment and label helmets in video footage

## Project setup

One can setup the project with the command (preferably in a virtual environment):
```bash
pip install -r requirements.txt
```

## Helmet detection notebook

The notebook `yolov5_helmet_detection.ipynb` displays an application of YOLOv5 to helmet detection. Running the whole notebook will generate a video clip with inference results obtained on one of the test videos. This video clip can be viewed in the last cell of the notebook.

Be careful to specify well where you store Kaggle's data for this challenge (in the second code cell of the notebook).

## Pipeline

The goal of this competition is to assign specific players to each helmet.

In order to do so, the targeted pipeline of this project is the following:
  1. Helmet detection with an object detection algorithm such as YOLOv5. There may be issues with sideline players' helmets.
  2. Helmet tracking so that several different instances do not switch with each other (SORT algorithm?). As there may be many players in a very small area, this can be challenging (occlusions).
  3. Use the provided tracking data to identify players:
     * Recognize teams (home and visitors). Usually the home team wears a white/clear jersey or the 3rd one more rarely.
     * Roles (QB, RB, WR, OL, etc...) can be well inferred using the positions of the players before the snap. This could enable to refine the search (for example, if we know that player X is a RB, it cannot be Mahomes). Furthermore, players' numbers are constrained by their role, so it can help to check the inferred label ({H, V}XX).
     * Pre-snap recognition of two different players playing at the same position can be tricky because their positions should be the same. Visual recognition can be performed but tricky for some players. Reading their numbers is probably the safest solution (if possible).
     * All players are visible from the sideline view ("all 22"), but not from the endzone one.
     * Map players between endzone view, sideline view and tracking data.
