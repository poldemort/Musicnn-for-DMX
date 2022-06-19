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

For instance,

![Temporal evolution of the classes](./dmx_Bastille.png "Tags")

Further [usage documentation is available on the Musicnn github](https://github.com/jordipons/musicnn).

## Algorithm performance

We plot the output of the network for the first 75 seconds of a song.

![Temporal evolution of the classes](./testdmx_Bastille_crop2.png "Tags")

We create the following output with our algorithm:

![Output](./dmx_switches_Bastille.png "Output")



## Lighting software

We use the lighting software [Sunlite](https://www.sunlitepro.com/en/sunlite.htm) we upload the scenes configuration in the file ![Show](./Show.zip "Show")

The configurated scenes have a keyboard letter assigned to be able to launch them from a Keyboard action generated by a Python script.



---

## Example

