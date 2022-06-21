# Musicnn-for-DMX
We use the ConvNet Musicnn for DMX lighting automation.

## Environment preparation

> We prepare the needed environment for the Musicnn network in a Ubuntu machine for simplicity. The versions of the packages must be python 3.7.13, tensorflow 1.15.0 and librosa 0.9.1. The network was created in these conditions and multiple methods are deprecated in actual versions. We replicated the environment on our Windows machine following the exported YAML file below.  It is possible that some components collision between them. We recommend to remove them from the YAML and install them manually later.

```
name: pythonDMX
channels:
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - ca-certificates=2022.4.26=haa95532_0
  - certifi=2022.5.18.1=py37haa95532_0
  - openssl=1.1.1o=h2bbff1b_0
  - pip=21.2.2=py37haa95532_0
  - python=3.7.13=h6244533_0
  - setuptools=61.2.0=py37haa95532_0
  - sqlite=3.38.3=h2bbff1b_0
  - vc=14.2=h21ff451_1
  - vs2015_runtime=14.27.29016=h5e58377_2
  - wheel=0.37.1=pyhd3eb1b0_0
  - wincertstore=0.2=py37haa95532_2
  - pip:
    - absl-py==1.0.0
    - appdirs==1.4.4
    - astor==0.8.1
    - audioread==2.1.9
    - cached-property==1.5.2
    - cffi==1.15.0
    - charset-normalizer==2.0.12
    - cycler==0.11.0
    - decorator==5.1.1
    - fonttools==4.33.3
    - gast==0.2.2
    - google-pasta==0.2.0
    - grpcio==1.44.0
    - h5py==3.6.0
    - idna==3.3
    - importlib-metadata==4.11.3
    - joblib==1.1.0
    - keras-applications==1.0.8
    - keras-preprocessing==1.1.2
    - kiwisolver==1.4.2
    - librosa==0.9.1
    - llvmlite==0.38.0
    - markdown==3.3.6
    - matplotlib==3.5.2
    - numba==0.55.1
    - numpy==1.21.6
    - opt-einsum==3.3.0
    - packaging==21.3
    - pillow==9.1.1
    - pooch==1.6.0
    - protobuf==3.20.1
    - pycparser==2.21
    - pynput==1.7.6
    - pyparsing==3.0.8
    - python-dateutil==2.8.2
    - requests==2.27.1
    - resampy==0.2.2
    - scikit-learn==0.22.1
    - scipy==1.7.3
    - six==1.16.0
    - soundfile==0.10.3.post1
    - tensorboard==1.15.0
    - tensorflow==1.15.0
    - tensorflow-estimator==1.15.1
    - termcolor==1.1.0
    - typing-extensions==4.2.0
    - urllib3==1.26.9
    - werkzeug==2.1.1
    - wrapt==1.14.0
    - zipp==3.8.0
prefix: C:\Users\Cousteau\anaconda3\envs\pythonDMX

```

## Network used

We employ the [Musicnn](https://arxiv.org/abs/1909.06654v1) network for predicting the temporal 50 tags. 

For instance, this is the analysis of the full song Pompeii by Bastille:

![Temporal evolution of the classes](./dmx_Bastille.png "Tags")

Further [usage documentation is available on the Musicnn github](https://github.com/jordipons/musicnn).

## Algorithm performance

We plot the output of the network for the first 75 seconds of a song and for just the first 27 tags. On the second figure, we plot the resulting switches. 

![Temporal evolution of the classes](./Bastille_final_crop.png "Tags")

The results of this execution are:
```
Minute 0, second 00.0: Switch: 1 (by default, low-arousal start configuration)
Minute 0, second 18.5: Switch: 5 (vocals changes)
Minute 0, second 32.5: Switch: 5 (vocals changes)
Minute 0, second 50.5: Switch: 3 (top-5 tags change)
Minute 1, second 00.5: Switch: 2 (total energy drop)
Minute 1, second 11.0: Switch: 6 (peak detection on tag 'rock')
```
Depending on the activated detector, a switch value is generated and the lighting scene will be launched accordingly. 


## Lighting software

We use the lighting software [Sunlite](https://www.sunlitepro.com/en/sunlite.htm) we upload the scenes configuration in the file ![Show](./Show.zip "Show").

The configurated scenes have a keyboard letter assigned to be able to launch them from a Keyboard action generated by a Python script.


---

## Operation example

It is important to have the lighting software correctly parametrized with the keyboard keys ready to launch lighting scenes. [The algorithm](./api_dmx_project.py) generates on-line keyboard press and release following the song's speed. Every half second the song's tags are evaluated and the scene's switch is possible. In the code:
```
for i in range(len(switches)):
  if switches[i] != 0:
    a = get_action(switches[i], last_action)
    last_action = a
    keyboard.press(str(a))
    keyboard.release(str(a))
  time.sleep(0.49)
```
So, if there is a scene switch at that timestep, the action is decided in `get_action` and the key is pressed to launch that scene. 
While the script is running, the keys might be activated, writing letters on our computer autonomously. The lighting software has to be in the foreground to receive these inputs. 

The script computes the full song at once and creates a countdown of ten seconds for the operator to switch its window to the lighting software. Once it starts, following the music speed, changes the scenes according to the algorithm's output. 

