# harmonics-demo
Interactive illustration of harmonics (overtones) and the importance of their phase for making a pulse train.

It is intended to be used to illustrate concepts related to ultrafast optics and attosecond pulse trains (high harmonic generation), but uses frequencies in a range that can be played back as sound. Sound playback is only available if the pyaudio package is installed in your Python environment.

## Screenshots
![Continuous-waveform example with odd and even harmonic orders and random phases](https://github.com/erik-mansson/harmonics-demo/blob/screenshots/screenshots/screenshot_1.png?raw=true)

![Odd-order harmonics with pi phase-shifts giving a pulse train with alternating signs](https://github.com/erik-mansson/harmonics-demo/blob/screenshots/screenshots/screenshot_2.png?raw=true)

![In the option to get an isolated pulse, an undocumented gating envelope function is applied and the sound playback makes longer pauses to hear the separate pulses](https://github.com/erik-mansson/harmonics-demo/blob/screenshots/screenshots/screenshot_3.png?raw=true)

## How to use it?

On Windows with Python via Anaconda or Miniconda, the easiest way to launch the program is to read and edit [harmonics_demo_launcher.bat](harmonics_demo_launcher.bat) and then double-click it create a shortcut to it. On Linux you run [harmonics_demo.py](harmonics_demo.py) like any python script, possibly after activating a suitable python environment.

The bottom left side of th graphical user interface has eight numeric input boxes for choosing the harmonic orders (1 is the fundamental frequency, 2 is twice that frequency and so on ...), sliders for choosing the amplitude of each tone, and knobs/dials for choosing their phase (0 is with the indicator to the right).

Above these controls are buttons to set them to some standard patterns, again arranged with a column for controlling the harmonic orders on the left, the amplitudes in the middle and the phases on the right.
The top left corner has some general settings, like whether to apply a temporal envelope or gating of the waveform, its fundamental frequency and whether to play it as sound (that slider controls the sound volume).

The rest of the window has diagrams. The white curves show waveforms of each individual tone as well as their sum (the final waveform, including the optional envelope function). On the right, the blue curve is the temporal intesity (square of the waveform) and the yellow curve is the spectral intensity i.e. the spectrum as per the Fast Fourier Transform (FFT). Finally the phase of each point in the spectrum is shown with small dots and larger dots connected by lines for peaks identified in the spectrum (to put less emphasis on all the random phases found from numeric noise in the zero-intensity part of the spectrum). When an envelope function is enabled (the top-left combobox), you will notice that the width of each peak in the spectrum broadens, and the phases where nearly zero intensity are less random.

## Dependencies
Python 3 environment with at least the packages numpy, PyQt5 and pyqtgraph. To be able to play the waveforms as audio, you also need the pyaudio package but the program runs without it.

