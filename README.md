# cpptopywrapper

cpptopywrapper contains all of the code for the object tracking pipeline for the AutoDrone team. 

## Installation

Clone the repository using git clone

```bash
git clone https://github.com/alexrog/cpptopywrapper.git
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```bash
pip install -r requirements.txt
```

## Usage

In the file `inference.py`, there exists the function `live_inference()` which gets the live camera feed and performs an inference on it. The output of the `nanodet.inference(frame)` is of type `BoundingBoxes` which is defined in `BoundingBoxes.py`. The information contained in this class can be used to publish the bounding box to a ROS node.

The `initialize_model` function is used to initialize the nanodet model. The parameters to this model are the file path to the nanodet model and the device to be used for inferencing: `CPU` or `MYRIAD`.

To run the live inferencing, ensure you are in the root directory of `cpptopywrapper` and run

```bash
python3 inference.py
```

Note: for live camera feed using `imshow` X forwarding must be enabled on ssh or you should be on the drone directly.