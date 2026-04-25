import io
from dataclasses import dataclass

import librosa
import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st


st.set_page_config(
    page_title="Fourier Audio Studio",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
:root {
    --bg: #f7f8fa;
    --surface: #ffffff;
    --ink: #0f172a;
    --muted: #475569;
    --line: #e2e8f0;
    --brand: #0f766e;
    --brand-soft: #ccfbf1;
    --accent: #f59e0b;
}

.stApp {
    background:
        radial-gradient(circle at 12% 8%, #d1fae5 0%, rgba(209, 250, 229, 0) 26%),
        radial-gradient(circle at 88% 20%, #fef3c7 0%, rgba(254, 243, 199, 0) 28%),
        linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
    color: var(--ink);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.hero {
    background: linear-gradient(120deg, #0f766e 0%, #115e59 52%, #78350f 100%);
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 18px;
    padding: 1.5rem 1.6rem;
    color: #ffffff;
    box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
}

.hero h1 {
    margin: 0;
    font-size: 1.9rem;
    line-height: 1.2;
    letter-spacing: 0.2px;
}

.hero p {
    margin: 0.55rem 0 0;
    color: rgba(255, 255, 255, 0.92);
}

.panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1rem;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
}

.metric-chip {
    display: inline-block;
    margin-right: 0.45rem;
    margin-bottom: 0.45rem;
    padding: 0.35rem 0.58rem;
    border-radius: 999px;
    font-size: 0.79rem;
    border: 1px solid #99f6e4;
    background: var(--brand-soft);
    color: #0f766e;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 10px;
    border: 1px solid #0f766e;
    background: linear-gradient(120deg, #0f766e 0%, #115e59 100%);
    color: white;
    font-weight: 600;
}

.stSlider [data-baseweb="slider"] div[role="slider"] {
    border-color: #115e59;
}
</style>
""",
    unsafe_allow_html=True,
)


@dataclass
class AudioBundle:
    signal: np.ndarray
    sr: int


def read_audio_from_upload(uploaded_file) -> AudioBundle:
    audio_bytes = uploaded_file.read()
    signal, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    signal = signal.astype(np.float64)

    peak = np.max(np.abs(signal))
    if peak > 1.0:
        signal = signal / peak

    return AudioBundle(signal=signal, sr=int(sr))


def audio_to_wav_bytes(signal: np.ndarray, sr: int) -> bytes:
    out = io.BytesIO()
    sf.write(out, signal.astype(np.float32), sr, format="WAV")
    out.seek(0)
    return out.read()


def compute_fft(signal: np.ndarray, sr: int):
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag = np.abs(fft_vals) / max(n, 1)
    phase = np.angle(fft_vals)
    return fft_vals, freqs, mag, phase


def apply_band_filter(fft_vals: np.ndarray, freqs: np.ndarray, low_hz: int, high_hz: int, mode: str):
    filtered = fft_vals.copy()
    mask = (freqs >= low_hz) & (freqs <= high_hz)

    if mode == "Band-stop (remove selected range)":
        filtered[mask] = 0
    else:
        filtered[~mask] = 0

    return filtered


def summarize_audio(signal: np.ndarray, sr: int, mag: np.ndarray, freqs: np.ndarray):
    duration = len(signal) / sr
    rms = float(np.sqrt(np.mean(np.square(signal))))

    if len(mag) > 1:
        dom_idx = np.argsort(mag[1:])[-5:][::-1] + 1
        dom_freqs = freqs[dom_idx]
    else:
        dom_freqs = np.array([0.0])

    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    centroid = float(np.mean(spectral_centroid))

    return {
        "duration": duration,
        "rms": rms,
        "centroid": centroid,
        "dominant": dom_freqs,
    }


def plot_waveform(signal: np.ndarray, sr: int, title: str):
    t = np.arange(len(signal)) / sr
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=signal,
            mode="lines",
            line=dict(width=1.1, color="#0f766e"),
            name="Amplitude",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_white",
        margin=dict(l=20, r=20, t=45, b=20),
        height=290,
    )
    return fig


def plot_spectrum(freqs: np.ndarray, mag: np.ndarray, title: str, sr: int):
    max_hz = min(sr // 2, 8000)
    mask = freqs <= max_hz

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=freqs[mask],
            y=mag[mask],
            mode="lines",
            line=dict(width=1.2, color="#b45309"),
            name="Magnitude",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Normalized Magnitude",
        template="plotly_white",
        margin=dict(l=20, r=20, t=45, b=20),
        height=290,
    )
    return fig


def plot_spectrogram(signal: np.ndarray, sr: int, title: str):
    stft = librosa.stft(signal, n_fft=2048, hop_length=512)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fig = go.Figure(
        data=go.Heatmap(
            z=db,
            colorscale="Viridis",
            colorbar=dict(title="dB"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Frame",
        yaxis_title="Frequency Bin",
        template="plotly_white",
        margin=dict(l=20, r=20, t=45, b=20),
        height=320,
    )
    return fig


def show_metrics(label: str, stats: dict):
    st.markdown(f"### {label}")
    st.markdown(
        f"""
<span class="metric-chip">Duration: {stats['duration']:.2f} s</span>
<span class="metric-chip">RMS Energy: {stats['rms']:.4f}</span>
<span class="metric-chip">Spectral Centroid: {stats['centroid']:.2f} Hz</span>
""",
        unsafe_allow_html=True,
    )

    dom = ", ".join([f"{f:.1f} Hz" for f in stats["dominant"][:5]])
    st.caption(f"Dominant components: {dom}")


def run_app():
    st.markdown(
        """
<div class="hero">
    <h1>Fourier Audio Studio</h1>
    <p>Upload audio, inspect waveform and spectrum, filter custom frequency bands, compare before vs after, then download the processed output.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    with st.sidebar:
        st.header("Control Panel")
        uploaded = st.file_uploader(
            "Upload audio",
            type=["wav", "mp3", "aac", "m4a", "flac", "ogg"],
            help="Supports common audio formats. The app converts output to WAV for reliable playback/download.",
        )

        filter_mode = st.radio(
            "Filter type",
            ["Band-stop (remove selected range)", "Band-pass (keep only selected range)"],
        )

        low_hz = st.slider("Low frequency (Hz)", min_value=0, max_value=20000, value=500, step=10)
        high_hz = st.slider("High frequency (Hz)", min_value=0, max_value=20000, value=3000, step=10)

        if low_hz > high_hz:
            low_hz, high_hz = high_hz, low_hz

        st.info(f"Selected band: {low_hz} Hz to {high_hz} Hz")
        process_clicked = st.button("Process", use_container_width=True)

    if uploaded is None:
        st.markdown('<div class="panel">Upload a file from the left panel to start analysis.</div>', unsafe_allow_html=True)
        return

    if not process_clicked:
        st.markdown('<div class="panel">Audio loaded. Click <b>Process</b> in the sidebar to start analysis and filtering.</div>', unsafe_allow_html=True)
        return

    original = read_audio_from_upload(uploaded)
    original_wav_bytes = audio_to_wav_bytes(original.signal, original.sr)

    fft_vals, freqs, mag, _ = compute_fft(original.signal, original.sr)
    original_stats = summarize_audio(original.signal, original.sr, mag, freqs)

    nyquist = original.sr // 2
    effective_low = max(0, min(low_hz, nyquist))
    effective_high = max(0, min(high_hz, nyquist))

    filtered_fft = apply_band_filter(fft_vals, freqs, effective_low, effective_high, filter_mode)
    filtered_signal = np.fft.irfft(filtered_fft, n=len(original.signal)).astype(np.float64)

    if np.max(np.abs(filtered_signal)) > 1.0:
        filtered_signal = filtered_signal / np.max(np.abs(filtered_signal))

    filtered_wav_bytes = audio_to_wav_bytes(filtered_signal, original.sr)

    f_fft, f_freqs, f_mag, _ = compute_fft(filtered_signal, original.sr)
    filtered_stats = summarize_audio(filtered_signal, original.sr, f_mag, f_freqs)

    mse = float(np.mean((original.signal - filtered_signal) ** 2))

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Step 1: Original Audio")
    st.audio(original_wav_bytes, format="audio/wav")
    show_metrics("Original signal metrics", original_stats)
    st.plotly_chart(plot_waveform(original.signal, original.sr, "Original Waveform"), use_container_width=True)
    st.plotly_chart(plot_spectrum(freqs, mag, "Original Magnitude Spectrum", original.sr), use_container_width=True)
    st.plotly_chart(plot_spectrogram(original.signal, original.sr, "Original Spectrogram"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Step 2: Filtered Audio")
    st.caption(
        f"Mode: {filter_mode} | Applied range: {effective_low} Hz to {effective_high} Hz | Nyquist: {nyquist} Hz"
    )
    st.audio(filtered_wav_bytes, format="audio/wav")
    show_metrics("Filtered signal metrics", filtered_stats)
    st.plotly_chart(plot_waveform(filtered_signal, original.sr, "Filtered Waveform"), use_container_width=True)
    st.plotly_chart(plot_spectrum(f_freqs, f_mag, "Filtered Magnitude Spectrum", original.sr), use_container_width=True)
    st.plotly_chart(plot_spectrogram(filtered_signal, original.sr, "Filtered Spectrogram"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Step 3: Before vs After Comparison")
        st.write(f"Mean Squared Error between original and filtered signals: {mse:.8f}")

        compare_fig = go.Figure()
        compare_len = min(len(original.signal), int(original.sr * 2))
        t = np.arange(compare_len) / original.sr
        compare_fig.add_trace(go.Scatter(x=t, y=original.signal[:compare_len], mode="lines", name="Original", line=dict(width=1.1, color="#0f766e")))
        compare_fig.add_trace(go.Scatter(x=t, y=filtered_signal[:compare_len], mode="lines", name="Filtered", line=dict(width=1.1, color="#dc2626")))
        compare_fig.update_layout(
            title="Waveform Comparison (first 2 seconds)",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            template="plotly_white",
            margin=dict(l=20, r=20, t=45, b=20),
            height=320,
        )
        st.plotly_chart(compare_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Step 4: Download Output")
        st.download_button(
            label="Download Filtered WAV",
            data=filtered_wav_bytes,
            file_name="filtered_output.wav",
            mime="audio/wav",
            use_container_width=True,
        )
        st.caption("This file is ready for sharing and deployment demos.")
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    run_app()
