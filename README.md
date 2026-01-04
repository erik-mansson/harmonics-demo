# harmonics-demo
Interactive illustration of concepts for ultrafast optics and velocity map imaging:
1. Harmonics (overtones) in the frequcney domain and the importance of their phase for making a pulse train.
2. 2D and 3D views of angular distribution from a sum of spherical harmonics (atomic physics and light--matter interaction).

## 1. Harmonics in the frequency and time domains
The harmonics_demo.py is intended to be used to illustrate concepts related to ultrafast optics and attosecond pulse trains (high harmonic generation), but uses frequencies in a range that can be played back as sound. Sound playback is only available if the pyaudio package is installed in your Python environment.

### Screenshots
![Continuous-waveform example with odd and even harmonic orders and random phases](https://github.com/erik-mansson/harmonics-demo/blob/screenshots/screenshots/screenshot_1.png?raw=true)
Continuous-waveform example with odd and even harmonic orders and random phases.

![Odd-order harmonics with pi phase-shifts giving a pulse train with alternating signs](https://github.com/erik-mansson/harmonics-demo/blob/screenshots/screenshots/screenshot_2.png?raw=true)
Odd-order harmonics with pi phase-shifts giving a pulse train with alternating signs.

![In the option to get an isolated pulse, an undocumented gating envelope function is applied and the sound playback makes longer pauses to hear the separate pulses](https://github.com/erik-mansson/harmonics-demo/blob/screenshots/screenshots/screenshot_3.png?raw=true)
In the option to get an isolated pulse, an undocumented gating envelope function is applied and the sound playback makes longer pauses to hear the separate pulses.

### How to use it?

On Windows with Python via Anaconda or Miniconda, the easiest way to launch the program is to read and edit [harmonics_demo_launcher.bat](harmonics_demo_launcher.bat) and then double-click it create a shortcut to it. On Linux you run [harmonics_demo.py](harmonics_demo.py) like any python script, possibly after activating a suitable python environment.

The bottom left side of the graphical user interface has eight numeric input boxes for choosing the harmonic orders (1 is the fundamental frequency, 2 is twice that frequency and so on ...) of eight tones, sliders for choosing their amplitudes, and knobs/dials for choosing their phases (0 is with the marker to the right).

Above these controls are buttons to apply to some standard patterns, again arranged with a column for controlling the harmonic orders on the left, the amplitudes in the middle and the phases on the right. The top left corner has some general settings, like the fundamental frequency, whether to apply a temporal envelope or gating of the waveform, and whether to play the waveform as sound (that slider controls the sound volume).

The rest of the window has diagrams showing the result in different ways. The white curves show waveforms of each individual tone as well as their sum (the final waveform, including the optional envelope function). On the right, the blue curve is the temporal intesity (square of the waveform) and the yellow curve is the spectral intensity i.e. the spectrum as per the Fast Fourier Transform (FFT). Finally the phase of each point in the spectrum is shown with small dots, as well as some larger dots connected by lines for peaks identified in the spectrum (to make them stand out from all the random phases found in the numeric noise at points in the spectrum with nearly zero intensity). When an envelope function is enabled (the top-left combobox), you will notice that the width of each peak in the spectrum increases, and the phase becomes less random even where nearly zero intensity.

### Dependencies
Python 3 environment with at least the packages numpy, PyQt5 and pyqtgraph. To be able to play the waveforms as audio, you also need the pyaudio package but the program runs without it.


## 2. Angular distribution projections
The angular_distribution_demo.py is intended to be used to illustrate concepts related to velocity map imaging, mainly of electrons from the photoioinzation of atoms where the result can typically be represented by the sum of a few [spherical harmonics](https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Visualization_of_complex_spherical_harmonics).

### Screenshots
<img width="484" height="352" alt="default s_0 p_1 projection" src="/../screenshots/screenshots/default%20s_0%20p_1%20projection.png" /><br/>The default example ("s<sub>0</sub>" + "p<sub>1</sub>") with narrow spheres.

<img width="484" height="352" alt="d_1 + f_2 projection" src="/../screenshots/screenshots/d_1%20%2B%20g_2%20projection.png" /><br/>An example with "d<sub>1</sub>" + "g<sub>2</sub>" with thick/blurry spheres slightly overlapping, projected onto an xz-plane (integrating intensity along the y axis).<br/>
<img width="358" height="215" alt="d_1 + f_2 slice abs, cropped" src="/../screenshots/screenshots/d_1%20%2B%20g_2%20slice%20abs%2C%20cropped.png" />&nbsp;<img width="358" height="215" alt="d_1 + f_2 phi,theta, cropped" src="/../screenshots/screenshots/d_1%20%2B%20g_2%20phi%2Ctheta%2C%20cropped.png" /><br/>Alternative visualizations of "d<sub>1</sub>" + "g<sub>2</sub>": (left) absolute values at the y=0 slice or (right) projection onto a sphere (integrating intensity radially).


## License
These programs and their documentation are released under the [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/), [summarized here](https://choosealicense.com/licenses/mpl-2.0/). You are free to distribute and develop them further under the terms of the license. If you use it for teaching, a link here would be appreciated (and would be a simple way of letting students download and play with the sofware themselves). If you wish to contribute improvements, click the Issues-tab and create an issue to describe your suggestion (or a pull-request with your version).

