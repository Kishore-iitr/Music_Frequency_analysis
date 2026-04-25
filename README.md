# Fourier Audio Studio

### Fourier Series & Fourier Transform Analysis of Real Audio

An interactive Streamlit-based audio analysis application that demonstrates the principles of Fourier Series, Fourier Transform, and signal reconstruction on real-world audio signals.

This project enables users to decompose audio into frequency components, manipulate them, and reconstruct the signal, providing both theoretical understanding and practical implementation of frequency-domain processing.

Link for the APP: LINK(https://musicfrequencyanalysis.streamlit.app/)


<img width="1914" height="865" alt="image" src="https://github.com/user-attachments/assets/bb90a213-9aa8-4f32-8ef6-614b843587b1" />


---

## Features

* Upload audio files (`.wav`, `.mp3`, `.aac`, `.flac`, etc.)
* Visualize:

  * Time-domain waveform
  * Frequency spectrum (FFT)
  * Spectrogram (STFT)
* Apply frequency filtering:

  * Band-stop (remove selected frequencies)
  * Band-pass (retain selected frequencies)
* Reconstruct signal using inverse FFT
* Compare original vs processed signal
* Compute reconstruction error (MSE)
* Play and download processed audio

---

## Concepts Covered

### Fourier Series

Any periodic signal can be expressed as a sum of sinusoids:
[
x(t) = \sum_{n=-\infty}^{\infty} C_n e^{j n \omega t}
]

The coefficients ( C_n ) represent the amplitude and phase of individual frequency components.

---

### Fourier Transform

Transforms a signal from time domain to frequency domain:
[
X(f) = \int x(t)e^{-j2\pi ft} dt
]

This shows how much of each frequency is present in the signal.

---

### Discrete Fourier Transform (DFT)

Used for sampled signals:
[
X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}
]

Efficiently implemented using the Fast Fourier Transform (FFT).

---

### Relation: Fourier Series vs DFT

| Fourier Series             | DFT                          |
| -------------------------- | ---------------------------- |
| Continuous periodic signal | Discrete sampled signal      |
| Infinite harmonics         | Finite frequency bins        |
| Analytical representation  | Computational implementation |

DFT serves as a practical approximation of the Fourier Transform for real-world discrete signals.

---

## How It Works

### Step 1: Audio Input

* Audio is loaded using `librosa`
* Converted to mono and normalized

### Step 2: Fourier Decomposition

```python
fft_vals = np.fft.rfft(signal)
```

Outputs:

* Frequencies
* Magnitude
* Phase

---

### Step 3: Frequency Filtering

* Band-stop: removes selected frequency range
* Band-pass: retains only selected frequency range

```python
filtered_fft[mask] = 0
```

---

### Step 4: Signal Reconstruction

```python
filtered_signal = np.fft.irfft(filtered_fft)
```

---

### Step 5: Error Analysis

Mean Squared Error:
[
MSE = \frac{1}{N} \sum (x[n] - \hat{x}[n])^2
]

---

## Visualizations

* Waveform (time domain)
* Frequency spectrum
* Spectrogram (time-frequency representation)
* Before vs after comparison

---

## User Interface

Built using Streamlit with:

* Interactive controls for frequency selection
* Real-time processing
* Audio playback
* Download functionality

---

## Project Structure

```
├── fourier_audio_studio_app.py
├── Fourier_Audio_Analysis_FULL.ipynb
├── requirements.txt
├── sample-12s.wav
└── README.md
```

---

## Installation and Setup

### Clone the repository

```bash
git clone https://github.com/your-username/fourier-audio-studio.git
cd fourier-audio-studio
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run fourier_audio_studio_app.py
```



## Key Outcomes

* Established the relation between Fourier Series and DFT
* Decomposed real audio into frequency components
* Reconstructed signals using inverse FFT
* Quantified reconstruction error
* Built an interactive frequency manipulation interface

---



Kishore
