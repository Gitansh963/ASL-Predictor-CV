# ASL Predictor

ASL Predictor is an open-source application that leverages computer vision and deep learning to recognize American Sign Language (ASL) gestures from live webcam feed. This application can be a useful tool for learning and practicing ASL, and it can also facilitate communication with individuals who are deaf or hard-of-hearing.

## Installation and Usage

To use this application, you need to download the data, the model, and the scripts. You also need to install the required packages.

### Prerequisites

- Download the data from the American Sign Language [Dataset](https://www.kaggle.com/datasets/prathumarikeri/american-sign-language-09az).
- Download the model from American [Model](https://firebasestorage.googleapis.com/v0/b/american-sign-language-7e378.appspot.com/o/american_model.h5?alt=media&token=7553a597-fa46-4228-a3f2-199086fecbf2)
- Clone the ASL Predictor Repository to your local machine.
- Install the required packages by running the following command in your anaconda terminal:

```bash
conda create --name <env> --file requirements.txt
```
### Running the Application
-Run the data_american.py script to preprocess the data and create the labels.
-Run the second.py script to build and train the model. Alternatively, you can load the pretrained model from the american_model.h5 file.
-Run the three.py script to launch the application. A window will open displaying your webcam feed and the predicted ASL gesture.
