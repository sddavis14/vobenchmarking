# vobenchmarking
This repository is for the CSC 591/791 Software for Robotics project. 

<b>vobenchmarking</b> is a software for the evaluation of the visual odometry algorithms RTABMAP and UnDeepVO. It also provides a custom implementation of the UnDeepVO algorithm. The evaluation can be done for different datasets, namely, 4seasons dataset and Tartan Air dataset.

<h2> Code Structure:</h2>
<ul>
<li> <b>dataset_processing/</b> - Contains the code for converting datasets to ROS bags.
<li> <b>ros2_ws/</b> - The workspace which contains the launch files to launch ROS nodes for different algorithms and datasets:
<ul>
<li> RTABMAP for 4Seasons dataset (rtab_4seasons_launch.py).
<li> RTABMAP for Tartan Air dataset (rtab_tartan_air_launch.py).
<li> UnDeepVO for 4Seasons dataset (undeep_4seasons_launch.py).
<li> UnDeepVO for Tartan Air dataset (undeep_tartan_air_launch.py).
</ul>
<li> <b>undeepvo/</b> - Code implementation of the UnDeepVO algorithm.
<li> <b>evaluator/</b> - Code for evaluation of the results generated after running the algorithm.
</ul>

<h2> Installation Instructions :</h2>

<h3> Requirements:</h3>
1) Ubuntu Jammy (22.04), Python<br>
2) Install <b>ROS2 Iron</b> (Desktop Install)- https://docs.ros.org/en/iron/Installation/Alternatives/Ubuntu-Install-Binary.html<br>
3) Install dependencies for Ubuntu:

```commandline
sudo apt install ros-iron-rtabmap* ros-iron-rtabmap-odom ros-iron-rtabmap-msgs ros-iron-image-transport-plugins
```
<br>
4) Install python dependencies:

```commandline
pip3 install torch torchvision kornia transforms3d pillow numpy
```
<br>
<h3>Clone this repository:</h3>

```commandline
git clone https://github.com/sddavis14/vobenchmarking.git
cd vobenchmarking
```
<br>
<h3> Creating ROS bag files from datasets:</h3>
<h4>Downloading Datasets:</h4>
Although the <b>/dataset_processing</b> folder contains samples as ROS bag files which require git lfs to download,
the complete datasets are available at:
<ul>
<li> 4Seasons: <a href="https://www.4seasons-dataset.com/dataset"> Download both images and ground truth pose.</a>
<li> TartanAir: <a href="https://theairlab.org/tartanair-dataset/"> Follow the instruction given on this link.</a>
</ul>

For 4Seasons, both images and ground truth pose must be downloaded and the GNSSPoses.txt must be added to the dataset folder of the image files.

<h4>Generating ROS bags:</h4>
Once the dataset has been downloaded, place it in the main project directory.

````commandline
cp /path/to/downloaded/dataset .
````
OR
````commandline
mv /path/to/downloaded/dataset .
````

The files 4seasons_to_bag.py and tartan_air_to_bag.py convert the datasets to bag files.

<b>4Seasons:</b>

```commandline
python3 dataset_processing/4seasons_to_bag.py
```

<b>TartanAir:</b>

```commandline
python3 dataset_processing/tartan_air_to_bag.py
```

<br>
<h3> Launching ROS nodes:</h3>

<h4> Setup the overlay:</h4>
In order to launch the ROS files, install the overlay.

```commandline
source ros2_ws/install/setup.bash
```

<h4> Launching:</h4>
Launch any one of the following at a time.

For 4Seasons and RTABMAP run:

```commandline
ros2 launch vo_eval_ros rtab_4seasons_launch.py
```

For Tartan Air and RTABMAP run:

```commandline
ros2 launch vo_eval_ros rtab_tartan_air_launch.py
```

For 4Seasons and UnDeepVO run: 

```commandline
ros2 launch vo_eval_ros undeep_4seasons_launch.py
```

For Tartan Air and UnDeepVO run:

```commandline
ros2 launch vo_eval_ros undeep_tartan_air_launch.py
```


<h4> Playing the ROS bag files:</h4>
Once the launching of any one of the above files is complete,
play the ROS bag.

```commandline
ros2 bag play /path/to/bagfile
```

This plays the bag file and the odometry node processes the data to store the results internally.

<h3> Evaluate the results:</h3>

You can evaluate the results by going into the 'evaluator/' directory and running the evaluator script:

```commandline
cd evaluator/eval
python3 evaluator.py
```

This will generate plots which can now be viewed in the 'evaluator/plots' directory:

```commandline
cd ../plots
```
