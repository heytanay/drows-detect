# Drowsines Detector v1
## About
This idea of this project came to me when searching throught the web and stumbling upon a random search statistic about Accidents caused by the Sleeping.

## What it does
This Program works by detecting faces, finding facial landmarks on them, extracting corrdinates of both eyes, calculating the Eye-aspect-ratio and then determining whether the eye is closed on not. If the eye is closed for long enough, it will start the alarm on a seperate thread and then put an alert on screen.
![Normal Mode](https://github.com/heytanay/drows-detect/blob/master/147.png)
![Drowsiness Alert Mode](https://github.com/heytanay/drows-detect/blob/master/148.png)

## Inspiration
I have implemented all of this by looking at the Amazing Code-along by Adrian at PyImageSearch. I have just understood and implemented that.

## Running the Code
To run the code on your standard Python 3.6 Installation, you must install the dependencies by following this:

```
pip install cmake

pip install scipy numpy matplotlib scikit-learn 

pip install opencv-contrib-python

pip install dlib

pip install playsound
```

Note: Make sure to install the libraries in above order only.

Now just, download the source code and then inside the directory, run this command:

For Windows 
```
python main.py
```

For MacOS and Linux:
```
sudo python3 main.py
```
