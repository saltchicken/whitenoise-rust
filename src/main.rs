/// # noise
///
/// A continuous Brown noise generator with optional bidirectional spectral
/// oscillation that plays to your default audio output device until you quit.
///
/// Controls:
/// * `Up` / `Down`    — increase / decrease volume (5% steps).
/// * `Left` / `Right` — decrease / increase oscillation cycle length (5s steps).
/// * `+` / `-`        — increase / decrease oscillation range away from Brown (5% steps).
/// * `o` or `Space`   — toggle oscillation on / off.
/// * `r`              — toggle reverb on / off.
/// * `[` / `]`        — decrease / increase reverb wet mix (5% steps).
/// * `q` or `Esc`     — quit.
/// * `Ctrl+C`         — quit.
use std::io::{Write, stdout};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI8, AtomicU8, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream};
use crossterm::Command;
use crossterm::cursor::{self, MoveTo};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::style::{
    Attribute, Color, ContentStyle, Print, SetAttribute, SetForegroundColor, SetStyle,
};
use crossterm::terminal::{self, Clear, ClearType};

/// A small, allocation-free Brown noise generator with smooth positive/negative
/// spectral oscillation around a pure Brown-noise center.
///
/// An internal `xorshift64` PRNG feeds the Brown integrator. A complementary
/// three-band decomposition (`low + mid + high == brown`) shifts the spectrum
/// smoothly darker (`offset < 0`) or brighter (`offset > 0`) without altering
/// filter pole coefficients or introducing clicks.
struct NoiseGen {
    /// `xorshift64` PRNG state (must never be zero).
    state: u64,
    /// Brown-noise (AR(1)) integrator state.
    brown: f64,
    /// Low-band (~<300 Hz) filter state for spectral tilt.
    low: f64,
    /// Low+mid (~<1400 Hz) filter state for spectral tilt.
    low_mid: f64,
    /// One-pole coefficient for the ~300 Hz split.
    alpha_low: f64,
    /// One-pole coefficient for the ~1400 Hz split.
    alpha_mid: f64,
    /// One-pole coefficient for smoothing `offset` transitions (~30 ms).
    alpha_offset: f64,
    /// Current smoothed spectral offset in `[-1.0, 1.0]` (`0.0` = pure Brown).
    offset: f64,
    /// LFO phase in `[0.0, 1.0)`.
    phase: f64,
    /// Interleaved samples per second (`sample_rate * channels`).
    samples_per_sec: f64,
}

impl NoiseGen {
    fn new(samples_per_sec: f64) -> Self {
        // Seed from the wall clock so successive runs differ.
        let mut seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        if seed == 0 {
            seed = 0x9E37_79B9_7F4A_7C15;
        }
        let sps = samples_per_sec.max(1.0);
        let tau = std::f64::consts::TAU;
        NoiseGen {
            state: seed,
            brown: 0.0,
            low: 0.0,
            low_mid: 0.0,
            alpha_low: 1.0 - (-tau * 300.0 / sps).exp(),
            alpha_mid: 1.0 - (-tau * 1400.0 / sps).exp(),
            alpha_offset: 1.0 - (-1.0 / (0.030 * sps)).exp(),
            offset: 0.0,
            phase: 0.0,
            samples_per_sec: sps,
        }
    }

    /// One internal uniform random excitation sample in roughly `[-1, 1]`.
    fn prng(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        // Map the top 53 bits onto [-1, 1].
        ((x >> 11) as f64) / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    /// Brownian (red) noise: an integrated random walk, gain-normalised.
    fn brown(&mut self) -> f64 {
        let w = self.prng();
        self.brown = (self.brown + 0.02 * w) / 1.02;
        self.brown * 3.5
    }

    /// Current spectral offset in `[-1.0, 1.0]` (`0.0` = pure Brown).
    fn current_offset(&self) -> f64 {
        self.offset
    }

    /// Apply continuous positive/negative spectral tilt around pure Brown noise.
    ///
    /// At `offset == 0.0`, `g_low == g_mid == g_high == 1.0`, so the sum
    /// reconstructs `b` identically (`low + mid + high == b`).
    fn tilt(&mut self, b: f64, offset: f64) -> f64 {
        self.low += self.alpha_low * (b - self.low);
        self.low_mid += self.alpha_mid * (b - self.low_mid);
        let mid = self.low_mid - self.low;
        let high = b - self.low_mid;

        let u = offset.clamp(-1.0, 1.0);
        let (g_low, g_mid, g_high) = if u >= 0.0 {
            (0.50_f64.powf(u), 1.30_f64.powf(u), 2.30_f64.powf(u))
        } else {
            let d = -u;
            (1.20_f64.powf(d), 0.25_f64.powf(d), 0.04_f64.powf(d))
        };

        g_low * self.low + g_mid * mid + g_high * high
    }

    /// Produce one output sample (in ~`[-1, 1]`) for Brown noise with optional
    /// symmetric positive/negative oscillation around Brown.
    fn sample(&mut self, osc_enabled: bool, period_secs: usize, osc_amount: u8) -> f64 {
        let target_offset = if osc_enabled {
            let period_samples = (period_secs.max(1) as f64) * self.samples_per_sec;
            self.phase = (self.phase + 1.0 / period_samples) % 1.0;
            let amount = (osc_amount.min(100) as f64) / 100.0;
            amount * (self.phase * std::f64::consts::TAU).sin()
        } else {
            self.phase = 0.0;
            0.0
        };
        self.offset += self.alpha_offset * (target_offset - self.offset);

        let b = self.brown();
        self.tilt(b, self.offset)
    }
}

/// Damped Schroeder comb filter for the Freeverb-style reverb tank.
struct CombFilter {
    buf: Vec<f64>,
    idx: usize,
    filter_store: f64,
    damp1: f64,
    damp2: f64,
    feedback: f64,
}

impl CombFilter {
    fn new(len: usize, feedback: f64, damp: f64) -> Self {
        let len = len.max(1);
        CombFilter {
            buf: vec![0.0; len],
            idx: 0,
            filter_store: 0.0,
            damp1: damp,
            damp2: 1.0 - damp,
            feedback,
        }
    }

    #[inline]
    fn process(&mut self, input: f64) -> f64 {
        let output = self.buf[self.idx];
        self.filter_store = output * self.damp2 + self.filter_store * self.damp1;
        self.buf[self.idx] = input + self.filter_store * self.feedback;
        self.idx += 1;
        if self.idx >= self.buf.len() {
            self.idx = 0;
        }
        output
    }
}

/// Unity-gain Schroeder allpass filter for reverb diffusion.
struct AllpassFilter {
    buf: Vec<f64>,
    idx: usize,
    feedback: f64,
}

impl AllpassFilter {
    fn new(len: usize, feedback: f64) -> Self {
        let len = len.max(1);
        AllpassFilter {
            buf: vec![0.0; len],
            idx: 0,
            feedback,
        }
    }

    #[inline]
    fn process(&mut self, input: f64) -> f64 {
        let delayed = self.buf[self.idx];
        let v = input + self.feedback * delayed;
        self.buf[self.idx] = v;
        self.idx += 1;
        if self.idx >= self.buf.len() {
            self.idx = 0;
        }
        delayed - self.feedback * v
    }
}

/// Single-channel Freeverb-style tank (8 parallel damped combs -> 4 series allpasses)
/// with a gentle sub-bass high-pass filter on the tank input so Brown noise's
/// deep bass does not build up coherently in the comb filters.
struct ReverbChannel {
    combs: [CombFilter; 8],
    allpasses: [AllpassFilter; 4],
    alpha_hp: f64,
    hp_prev_x: f64,
    hp_y: f64,
}

impl ReverbChannel {
    // Classic Freeverb delay lengths at 44.1 kHz.
    const COMB_TUNINGS: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
    const ALLPASS_TUNINGS: [usize; 4] = [556, 441, 341, 225];
    const STEREO_SPREAD: usize = 23;
    const COMB_FEEDBACK: f64 = 0.85;
    const ALLPASS_FEEDBACK: f64 = 0.5;
    const COMB_SCALE: f64 = 1.0 / 6.0;

    fn new(sample_rate: f64, channel_idx: usize) -> Self {
        let sr = sample_rate.max(1.0);
        let scale = sr / 44_100.0;
        let spread = (channel_idx % 2) * Self::STEREO_SPREAD + (channel_idx / 2) * 11;
        let tau = std::f64::consts::TAU;
        let damp = (-tau * 4000.0 / sr).exp();
        let alpha_hp = (-tau * 45.0 / sr).exp();

        let scale_len =
            |base: usize| -> usize { (((base + spread) as f64) * scale).round().max(1.0) as usize };

        let combs = [
            CombFilter::new(scale_len(Self::COMB_TUNINGS[0]), Self::COMB_FEEDBACK, damp),
            CombFilter::new(scale_len(Self::COMB_TUNINGS[1]), Self::COMB_FEEDBACK, damp),
            CombFilter::new(scale_len(Self::COMB_TUNINGS[2]), Self::COMB_FEEDBACK, damp),
            CombFilter::new(scale_len(Self::COMB_TUNINGS[3]), Self::COMB_FEEDBACK, damp),
            CombFilter::new(scale_len(Self::COMB_TUNINGS[4]), Self::COMB_FEEDBACK, damp),
            CombFilter::new(scale_len(Self::COMB_TUNINGS[5]), Self::COMB_FEEDBACK, damp),
            CombFilter::new(scale_len(Self::COMB_TUNINGS[6]), Self::COMB_FEEDBACK, damp),
            CombFilter::new(scale_len(Self::COMB_TUNINGS[7]), Self::COMB_FEEDBACK, damp),
        ];

        let allpasses = [
            AllpassFilter::new(scale_len(Self::ALLPASS_TUNINGS[0]), Self::ALLPASS_FEEDBACK),
            AllpassFilter::new(scale_len(Self::ALLPASS_TUNINGS[1]), Self::ALLPASS_FEEDBACK),
            AllpassFilter::new(scale_len(Self::ALLPASS_TUNINGS[2]), Self::ALLPASS_FEEDBACK),
            AllpassFilter::new(scale_len(Self::ALLPASS_TUNINGS[3]), Self::ALLPASS_FEEDBACK),
        ];

        ReverbChannel {
            combs,
            allpasses,
            alpha_hp,
            hp_prev_x: 0.0,
            hp_y: 0.0,
        }
    }

    #[inline]
    fn process_wet(&mut self, input: f64) -> f64 {
        let hp = self.alpha_hp * (self.hp_y + input - self.hp_prev_x);
        self.hp_prev_x = input;
        self.hp_y = hp;

        let mut acc = 0.0;
        for comb in &mut self.combs {
            acc += comb.process(hp);
        }
        let mut out = acc * Self::COMB_SCALE;
        for allpass in &mut self.allpasses {
            out = allpass.process(out);
        }
        out
    }
}

/// Multi-channel stereo-spread reverb with click-free mix smoothing and
/// equal-power dry/wet crossfade.
struct Reverb {
    channels: Vec<ReverbChannel>,
    alpha_mix: f64,
    mix: f64,
    dry_gain: f64,
    wet_gain: f64,
}

impl Reverb {
    fn new(sample_rate: f64, num_channels: usize) -> Self {
        let sr = sample_rate.max(1.0);
        let n = num_channels.max(1);
        let channels = (0..n).map(|ch| ReverbChannel::new(sr, ch)).collect();
        Reverb {
            channels,
            alpha_mix: 1.0 - (-1.0 / (0.030 * sr)).exp(),
            mix: 0.0,
            dry_gain: 1.0,
            wet_gain: 0.0,
        }
    }

    /// Advance the smoothed wet mix once per audio frame so all channels share
    /// the exact same crossfade gains.
    #[inline]
    fn begin_frame(&mut self, enabled: bool, mix_pct: u8) {
        let target = if enabled {
            (mix_pct.min(100) as f64) / 100.0
        } else {
            0.0
        };
        self.mix += self.alpha_mix * (target - self.mix);
        let angle = self.mix * std::f64::consts::FRAC_PI_2;
        self.dry_gain = angle.cos();
        self.wet_gain = angle.sin();
    }

    /// Process one sample for `channel_idx` using the current frame's mix gains.
    #[inline]
    fn process_channel(&mut self, channel_idx: usize, dry: f64) -> f64 {
        let wet = self.channels[channel_idx].process_wet(dry);
        self.dry_gain * dry + self.wet_gain * wet
    }
}

/// Shared lock-free parameters between the UI loop and the audio callback.
struct AudioParams {
    volume: AtomicU8,
    osc_enabled: AtomicBool,
    osc_secs: AtomicUsize,
    osc_amount: AtomicU8,
    osc_pos: AtomicI8,
    reverb_enabled: AtomicBool,
    reverb_mix: AtomicU8,
}

impl AudioParams {
    fn new() -> Self {
        AudioParams {
            volume: AtomicU8::new(20),
            osc_enabled: AtomicBool::new(true),
            osc_secs: AtomicUsize::new(30), // 30s full oscillation cycle
            osc_amount: AtomicU8::new(25),  // ±25% away from Brown
            osc_pos: AtomicI8::new(0),      // current position in -100..=100%
            reverb_enabled: AtomicBool::new(true),
            reverb_mix: AtomicU8::new(35), // 35% wet mix
        }
    }
}

/// Build a playing output `Stream` for the requested device.
fn create_stream(device: &Device, params: Arc<AudioParams>) -> Result<Stream, String> {
    let config = device
        .default_output_config()
        .map_err(|e| format!("no output config: {e}"))?;
    let sample_rate = config.sample_rate().0 as f64;
    let channels = config.channels().max(1) as usize;
    let samples_per_sec = sample_rate * channels as f64;

    let stream = match config.sample_format() {
        SampleFormat::F32 => {
            let mut r#gen = NoiseGen::new(samples_per_sec);
            let mut reverb = Reverb::new(sample_rate, channels);
            device
                .build_output_stream(
                    &config.config(),
                    move |buffer: &mut [f32], _| {
                        let gain = params.volume.load(Ordering::Relaxed) as f32 / 100.0;
                        let enabled = params.osc_enabled.load(Ordering::Relaxed);
                        let period = params.osc_secs.load(Ordering::Relaxed);
                        let amount = params.osc_amount.load(Ordering::Relaxed);
                        let rev_enabled = params.reverb_enabled.load(Ordering::Relaxed);
                        let rev_mix = params.reverb_mix.load(Ordering::Relaxed);
                        for frame in buffer.chunks_mut(channels) {
                            reverb.begin_frame(rev_enabled, rev_mix);
                            for (ch, s) in frame.iter_mut().enumerate() {
                                let dry = r#gen.sample(enabled, period, amount);
                                let mixed = reverb.process_channel(ch, dry);
                                *s = (mixed as f32 * gain).clamp(-1.0, 1.0);
                            }
                        }
                        let pos_pct = (r#gen.current_offset() * 100.0).round() as i8;
                        params.osc_pos.store(pos_pct, Ordering::Relaxed);
                    },
                    |err| eprintln!("audio stream error: {err}"),
                    None,
                )
                .map_err(|e| e.to_string())?
        }
        SampleFormat::I16 => {
            let mut r#gen = NoiseGen::new(samples_per_sec);
            let mut reverb = Reverb::new(sample_rate, channels);
            device
                .build_output_stream(
                    &config.config(),
                    move |buffer: &mut [i16], _| {
                        let gain = params.volume.load(Ordering::Relaxed) as f64 / 100.0;
                        let enabled = params.osc_enabled.load(Ordering::Relaxed);
                        let period = params.osc_secs.load(Ordering::Relaxed);
                        let amount = params.osc_amount.load(Ordering::Relaxed);
                        let rev_enabled = params.reverb_enabled.load(Ordering::Relaxed);
                        let rev_mix = params.reverb_mix.load(Ordering::Relaxed);
                        for frame in buffer.chunks_mut(channels) {
                            reverb.begin_frame(rev_enabled, rev_mix);
                            for (ch, s) in frame.iter_mut().enumerate() {
                                let dry = r#gen.sample(enabled, period, amount);
                                let v = reverb.process_channel(ch, dry) * gain * 32_767.0;
                                *s = v.clamp(-32_768.0, 32_767.0) as i16;
                            }
                        }
                        let pos_pct = (r#gen.current_offset() * 100.0).round() as i8;
                        params.osc_pos.store(pos_pct, Ordering::Relaxed);
                    },
                    |err| eprintln!("audio stream error: {err}"),
                    None,
                )
                .map_err(|e| e.to_string())?
        }
        other => {
            return Err(format!(
                "unsupported sample format from audio backend: {other}"
            ));
        }
    };

    stream
        .play()
        .map_err(|e| format!("failed to start audio: {e}"))?;
    Ok(stream)
}

/// Format a signed percentage in `-100..=100` to an exact 5-character string.
fn format_signed_pct(v: i8) -> String {
    let clamped = v.clamp(-100, 100);
    if clamped == 0 {
        "   0%".to_string()
    } else {
        format!("{:>+4}%", clamped)
    }
}

/// A single cell of text (optionally bold) within a row.
struct Cell {
    text: String,
    bold: bool,
    color: Color,
}

/// A row of the status panel: a bold label, a value, and (optionally) a
/// right-aligned accent, separated by fixed gaps. The row is padded to the
/// live terminal width so the right-hand border always lands on the same
/// column, no matter how wide or narrow the viewport is.
struct Row {
    label: String,
    value: String,
    bold_value: bool,
    value_color: Color,
    accent: Option<String>,
    accent_bold: bool,
    accent_color: Color,
}

impl Row {
    /// Render the row as `[left-pad][label][gap][value][gap][right-pad][accent][right-margin]`.
    ///
    /// `inner_width` is the box interior between the left and right `│`
    /// borders (`terminal_width - 2`). Labels are padded to `label_width` so
    /// every row's value column starts at the same column, right-aligned
    /// accents sit 2 columns inside the right border, and any overflow on a
    /// narrow terminal is truncated so the row never wraps past the border.
    fn cells(&self, inner_width: usize, label_width: usize) -> Vec<Cell> {
        const LEFT_PAD: &str = "  ";
        const GAP1: &str = "  ";
        const GAP2: &str = "  ";
        const RIGHT_MARGIN: &str = "  ";

        let label_text = format!("{:<width$}", self.label, width = label_width);
        let accent_w = self.accent.as_ref().map_or(0, |a| a.chars().count());
        let gap2_w = if self.accent.is_some() {
            GAP2.chars().count()
        } else {
            0
        };

        let overhead = LEFT_PAD.chars().count()
            + label_text.chars().count()
            + GAP1.chars().count()
            + gap2_w
            + accent_w
            + RIGHT_MARGIN.chars().count();

        let avail_for_value = inner_width.saturating_sub(overhead);
        let value_w = self.value.chars().count();
        let value_text = if value_w <= avail_for_value {
            self.value.clone()
        } else if avail_for_value >= 1 {
            let mut s: String = self
                .value
                .chars()
                .take(avail_for_value.saturating_sub(1))
                .collect();
            if avail_for_value >= 2 {
                s.push('…');
            } else {
                s = self.value.chars().take(1).collect();
            }
            s
        } else {
            String::new()
        };

        let used = overhead + value_text.chars().count();
        let right_pad = inner_width.saturating_sub(used);

        let mut cells = vec![
            Cell {
                text: LEFT_PAD.to_string(),
                bold: false,
                color: Color::Grey,
            },
            Cell {
                text: label_text,
                bold: true,
                color: Color::White,
            },
            Cell {
                text: GAP1.to_string(),
                bold: false,
                color: self.value_color,
            },
            Cell {
                text: value_text,
                bold: self.bold_value,
                color: self.value_color,
            },
        ];
        if right_pad > 0 {
            cells.push(Cell {
                text: " ".repeat(right_pad),
                bold: false,
                color: self.value_color,
            });
        }
        if let Some(accent) = &self.accent {
            cells.push(Cell {
                text: GAP2.to_string(),
                bold: false,
                color: self.value_color,
            });
            cells.push(Cell {
                text: accent.clone(),
                bold: self.accent_bold,
                color: self.accent_color,
            });
        }
        cells.push(Cell {
            text: RIGHT_MARGIN.to_string(),
            bold: false,
            color: Color::Grey,
        });

        // Guarantee total character count never exceeds `inner_width`, even on
        // ultra-narrow terminals where `overhead > inner_width`.
        let mut remaining = inner_width;
        let mut trimmed = Vec::with_capacity(cells.len());
        for mut cell in cells {
            if remaining == 0 {
                break;
            }
            let len = cell.text.chars().count();
            if len > remaining {
                cell.text = cell.text.chars().take(remaining).collect();
                remaining = 0;
            } else {
                remaining -= len;
            }
            trimmed.push(cell);
        }
        trimmed
    }
}

/// Append a crossterm `Command`'s ANSI escape sequence to `out`.
///
/// The `Command` trait isn't `dyn`-compatible (its `write_ansi` takes a
/// generic `impl fmt::Write`), so we can't keep heterogeneous commands in a
/// `Vec<Box<dyn Command>>`; serializing each one into one string avoids that
/// and also collapses the writes into a single stdout syscall.
fn push_command<C: Command>(out: &mut String, cmd: C) -> std::io::Result<()> {
    let mut buf = String::new();
    cmd.write_ansi(&mut buf).map_err(std::io::Error::other)?;
    out.push_str(&buf);
    Ok(())
}

/// Snapshot of the parameters rendered in the status panel.
#[derive(Clone, Copy, PartialEq, Eq)]
struct UiState {
    volume: u8,
    osc_enabled: bool,
    osc_secs: usize,
    osc_amount: u8,
    osc_pos: i8,
    reverb_enabled: bool,
    reverb_mix: u8,
}

impl UiState {
    fn from_params(params: &AudioParams) -> Self {
        let osc_enabled = params.osc_enabled.load(Ordering::Relaxed);
        let osc_pos = if osc_enabled {
            params.osc_pos.load(Ordering::Relaxed)
        } else {
            0
        };
        UiState {
            volume: params.volume.load(Ordering::Relaxed),
            osc_enabled,
            osc_secs: params.osc_secs.load(Ordering::Relaxed),
            osc_amount: params.osc_amount.load(Ordering::Relaxed),
            osc_pos,
            reverb_enabled: params.reverb_enabled.load(Ordering::Relaxed),
            reverb_mix: params.reverb_mix.load(Ordering::Relaxed),
        }
    }
}

/// Build the full ANSI frame for a terminal of `width` columns.
fn build_frame(width: usize, state: UiState) -> std::io::Result<String> {
    let UiState {
        volume,
        osc_enabled,
        osc_secs,
        osc_amount,
        osc_pos,
        reverb_enabled,
        reverb_mix,
    } = state;
    let inner = width.saturating_sub(2); // everything between left `│` and right `│`
    let border_inner = width.saturating_sub(2); // dash run between ┌ and ┐

    let filled = (volume as usize) / 5; // 0..=20 segments
    let mut bar = String::with_capacity(22);
    bar.push('│');
    for i in 0..20 {
        bar.push(if i < filled { '█' } else { '░' });
    }
    bar.push('│');

    let rev_filled = if reverb_enabled {
        (reverb_mix.min(100) as usize) / 5
    } else {
        0
    };
    let mut rev_bar = String::with_capacity(22);
    rev_bar.push('│');
    for i in 0..20 {
        rev_bar.push(if i < rev_filled { '█' } else { '░' });
    }
    rev_bar.push('│');

    let slider_width = 21usize; // index 10 = exact 0% Brown center
    let center = slider_width / 2;
    let clamped_pos = osc_pos.clamp(-100, 100) as i32;
    let pos = (((clamped_pos + 100) * (slider_width as i32 - 1) + 100) / 200) as usize;
    let amt = osc_amount.min(100) as i32;
    let min_pos = (((-amt + 100) * (slider_width as i32 - 1) + 100) / 200) as usize;
    let max_pos = (((amt + 100) * (slider_width as i32 - 1) + 100) / 200) as usize;

    let mut slider = String::with_capacity(slider_width + 4);
    slider.push_str("-[");
    for i in 0..slider_width {
        if (osc_enabled && i == pos) || (!osc_enabled && i == center) {
            slider.push('●');
        } else if i == center {
            slider.push('┼');
        } else if osc_enabled && i >= min_pos && i <= max_pos {
            slider.push('━');
        } else {
            slider.push('─');
        }
    }
    slider.push_str("]+");

    let mode_label = if osc_enabled {
        "Brown (oscillating)"
    } else {
        "Brown (steady)"
    };
    let osc_color = if osc_enabled {
        Color::Magenta
    } else {
        Color::DarkGrey
    };
    let range_str = format!("±{}%", osc_amount.min(100));
    let pos_str = format_signed_pct(if osc_enabled { osc_pos } else { 0 });

    // Model each row as data; the `Row` helper pads it to the live terminal
    // width so the right-hand border always lines up.
    let mode = Row {
        label: "mode".to_string(),
        value: mode_label.to_string(),
        bold_value: true,
        value_color: Color::Yellow,
        accent: None,
        accent_bold: false,
        accent_color: Color::White,
    };
    let osc = Row {
        label: "osc".to_string(),
        value: format!("{:>3}s  {:<5}  {slider}", osc_secs, range_str),
        bold_value: true,
        value_color: osc_color,
        accent: Some(pos_str),
        accent_bold: false,
        accent_color: Color::White,
    };
    let rev_color = if reverb_enabled {
        Color::Green
    } else {
        Color::DarkGrey
    };
    let rev_row = Row {
        label: "reverb".to_string(),
        value: format!("{rev_bar} {:>3}%", reverb_mix.min(100)),
        bold_value: true,
        value_color: rev_color,
        accent: Some(if reverb_enabled { " on" } else { "off" }.to_string()),
        accent_bold: reverb_enabled,
        accent_color: if reverb_enabled {
            Color::Green
        } else {
            Color::DarkGrey
        },
    };
    let volume_row = Row {
        label: "volume".to_string(),
        value: format!("{bar} {:>3}%", volume),
        bold_value: true,
        value_color: Color::Cyan,
        accent: None,
        accent_bold: false,
        accent_color: Color::White,
    };
    let controls = Row {
        label: "controls".to_string(),
        value: "↑/↓ vol  ←/→ cycle  +/- range  o osc  r/[/] rev  q quit".to_string(),
        bold_value: false,
        value_color: Color::DarkGrey,
        accent: None,
        accent_bold: false,
        accent_color: Color::White,
    };
    let rows = [mode, osc, rev_row, volume_row, controls];
    let label_width = rows
        .iter()
        .map(|r| r.label.chars().count())
        .max()
        .unwrap_or(0);

    // Top border with a centered title and a bottom border, both sized to the
    // live width (no hand-counted dashes).
    let title = " noise ";
    let title_w = title.chars().count();
    let top_inner = if border_inner >= title_w {
        let left_pad = (border_inner - title_w) / 2;
        let right_pad = border_inner - title_w - left_pad;
        format!("{}{}{}", "─".repeat(left_pad), title, "─".repeat(right_pad))
    } else {
        "─".repeat(border_inner)
    };
    let top = if width >= 2 {
        format!("┌{top_inner}┐")
    } else {
        String::new()
    };
    let bottom = if width >= 2 {
        format!("└{}┘", "─".repeat(border_inner))
    } else {
        String::new()
    };

    // Serialize every command (styling + text) into a single ANSI string and
    // write it in one shot. Use explicit `MoveTo(0, y)` per line so raw-mode
    // LF-without-CR and terminal pending-wrap at column `width` never misalign
    // subsequent rows.
    let mut out = String::with_capacity(1024);
    push_command(&mut out, MoveTo(0, 0))?;
    push_command(&mut out, SetStyle(ContentStyle::default()))?;
    push_command(&mut out, SetAttribute(Attribute::Bold))?;
    push_command(&mut out, SetForegroundColor(Color::White))?;
    push_command(&mut out, Print(top))?;
    push_command(&mut out, SetAttribute(Attribute::Reset))?;

    for (idx, row) in rows.iter().enumerate() {
        push_command(&mut out, MoveTo(0, (idx + 1) as u16))?;
        if width >= 2 {
            push_command(&mut out, SetStyle(ContentStyle::default()))?;
            push_command(&mut out, SetForegroundColor(Color::Grey))?;
            push_command(&mut out, Print("│"))?;
        }
        for c in row.cells(inner, label_width) {
            push_command(
                &mut out,
                if c.bold {
                    SetAttribute(Attribute::Bold)
                } else {
                    SetAttribute(Attribute::Reset)
                },
            )?;
            push_command(&mut out, SetForegroundColor(c.color))?;
            push_command(&mut out, Print(c.text))?;
        }
        if width >= 2 {
            push_command(&mut out, SetStyle(ContentStyle::default()))?;
            push_command(&mut out, SetForegroundColor(Color::Grey))?;
            push_command(&mut out, Print("│"))?;
        }
    }

    push_command(&mut out, MoveTo(0, (rows.len() + 1) as u16))?;
    push_command(&mut out, SetAttribute(Attribute::Bold))?;
    push_command(&mut out, SetForegroundColor(Color::White))?;
    push_command(&mut out, Print(bottom))?;
    push_command(&mut out, SetStyle(ContentStyle::default()))?;
    push_command(&mut out, MoveTo(0, (rows.len() + 2) as u16))?;
    push_command(&mut out, Clear(ClearType::FromCursorDown))?;

    Ok(out)
}

/// Paint the status panel describing the current state.
fn render(state: UiState) -> std::io::Result<()> {
    // Resolve the actual terminal width so the box always fits the real
    // viewport instead of a hard-coded ~65-column frame that overflows narrow
    // terminals or looks sparse on wide ones.
    let width = terminal::size().map(|(w, _)| w as usize).unwrap_or(80);
    let out = build_frame(width, state)?;
    let mut handle = stdout();
    handle.write_all(out.as_bytes())?;
    handle.flush()
}

/// Block until SIGINT, then flip `running` to false.
fn spawn_sigint_handler(running: Arc<AtomicBool>) {
    thread::spawn(move || {
        // `Signals::new` registers the handler; `forever()` blocks until one fires.
        match signal_hook::iterator::Signals::new([signal_hook::consts::SIGINT]) {
            Ok(mut signals) => {
                if signals.forever().next().is_some() {
                    running.store(false, Ordering::SeqCst);
                }
            }
            Err(e) => eprintln!("failed to install SIGINT handler: {e}"),
        }
    });
}

fn main() {
    // Shared state between the audio thread and the keyboard event loop.
    let running = Arc::new(AtomicBool::new(true));
    let params = Arc::new(AudioParams::new());

    // Start audio.
    let host = cpal::default_host();
    let device = match host.default_output_device() {
        Some(d) => d,
        None => {
            eprintln!("error: no audio output device found");
            std::process::exit(1);
        }
    };
    let stream = match create_stream(&device, params.clone()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    // Reliable Ctrl+C via SIGINT (covers terminals that still raise the signal).
    spawn_sigint_handler(running.clone());

    // Terminal setup.
    if let Err(e) = terminal::enable_raw_mode() {
        eprintln!("error: could not enable raw mode: {e}");
        std::process::exit(1);
    }
    let _ = execute!(stdout(), Clear(ClearType::All), cursor::Hide);

    // Initial paint.
    let mut last_state = UiState::from_params(&params);
    let _ = render(last_state);

    // Event loop: poll with a short timeout so we can also react to the
    // signal handler thread and animate the oscillation slider.
    loop {
        if !running.load(Ordering::Relaxed) {
            break;
        }

        let mut resized = false;
        match event::poll(Duration::from_millis(50)) {
            Ok(true) => match event::read() {
                Ok(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                    // Ctrl+C → quit (raw mode delivers it as a key event on
                    // most platforms; the SIGINT thread is the safety net).
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('c')
                    {
                        running.store(false, Ordering::SeqCst);
                        break;
                    }

                    match key.code {
                        KeyCode::Up => {
                            let v = params.volume.load(Ordering::Relaxed);
                            if v < 100 {
                                params
                                    .volume
                                    .store(v.saturating_add(5).min(100), Ordering::Relaxed);
                            }
                        }
                        KeyCode::Down => {
                            let v = params.volume.load(Ordering::Relaxed);
                            if v > 0 {
                                params.volume.store(v.saturating_sub(5), Ordering::Relaxed);
                            }
                        }
                        KeyCode::Right => {
                            let s = params.osc_secs.load(Ordering::Relaxed);
                            params.osc_secs.store((s + 5).min(300), Ordering::Relaxed);
                        }
                        KeyCode::Left => {
                            let s = params.osc_secs.load(Ordering::Relaxed);
                            params
                                .osc_secs
                                .store(s.saturating_sub(5).max(5), Ordering::Relaxed);
                        }
                        KeyCode::Char('+') | KeyCode::Char('=') => {
                            let a = params.osc_amount.load(Ordering::Relaxed);
                            params
                                .osc_amount
                                .store(a.saturating_add(5).min(100), Ordering::Relaxed);
                        }
                        KeyCode::Char('-') | KeyCode::Char('_') => {
                            let a = params.osc_amount.load(Ordering::Relaxed);
                            params
                                .osc_amount
                                .store(a.saturating_sub(5).max(5), Ordering::Relaxed);
                        }
                        KeyCode::Char('o') | KeyCode::Char('O') | KeyCode::Char(' ') => {
                            let next = !params.osc_enabled.load(Ordering::Relaxed);
                            params.osc_enabled.store(next, Ordering::Relaxed);
                        }
                        KeyCode::Char('r') | KeyCode::Char('R') => {
                            let next = !params.reverb_enabled.load(Ordering::Relaxed);
                            params.reverb_enabled.store(next, Ordering::Relaxed);
                        }
                        KeyCode::Char(']') => {
                            let m = params.reverb_mix.load(Ordering::Relaxed);
                            params
                                .reverb_mix
                                .store(m.saturating_add(5).min(100), Ordering::Relaxed);
                        }
                        KeyCode::Char('[') => {
                            let m = params.reverb_mix.load(Ordering::Relaxed);
                            params
                                .reverb_mix
                                .store(m.saturating_sub(5).max(5), Ordering::Relaxed);
                        }
                        KeyCode::Char('q') | KeyCode::Esc => {
                            running.store(false, Ordering::SeqCst);
                            break;
                        }
                        _ => {}
                    }
                }
                Ok(Event::Resize(_, _)) => {
                    resized = true;
                }
                // Ignore other events and key repeats/releases.
                _ => {}
            },
            Ok(false) => {}
            Err(e) => {
                eprintln!("input error: {e}");
                break;
            }
        }

        // Redraw only when something changed, to avoid flicker.
        let cur_state = UiState::from_params(&params);
        if resized || cur_state != last_state {
            if resized {
                let _ = execute!(stdout(), Clear(ClearType::All));
            }
            last_state = cur_state;
            let _ = render(cur_state);
        }
    }

    // Clean shutdown.
    let _ = terminal::disable_raw_mode();
    // Dropping the stream stops playback.
    drop(stream);

    // Wipe the status panel and restore the cursor so nothing is left on screen.
    let _ = execute!(stdout(), Clear(ClearType::All), MoveTo(0, 0), cursor::Show);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steady_brown_stays_at_zero_offset() {
        let mut r#gen = NoiseGen::new(96_000.0);
        for _ in 0..10_000 {
            let s = r#gen.sample(false, 30, 50);
            assert!(s.is_finite());
        }
        assert_eq!(r#gen.current_offset(), 0.0);
    }

    #[test]
    fn oscillation_reaches_positive_and_negative_range() {
        let sps = 1_000.0;
        let mut r#gen = NoiseGen::new(sps);
        let mut min_off = f64::INFINITY;
        let mut max_off = f64::NEG_INFINITY;
        // Run slightly over 1 full 2-second cycle.
        for _ in 0..2_200 {
            let s = r#gen.sample(true, 2, 60);
            assert!(s.is_finite());
            min_off = min_off.min(r#gen.current_offset());
            max_off = max_off.max(r#gen.current_offset());
        }
        assert!((0.55..=0.61).contains(&max_off), "max_off={max_off}");
        assert!((-0.61..=-0.55).contains(&min_off), "min_off={min_off}");
    }

    #[test]
    fn tilt_at_zero_is_exact_identity() {
        let mut r#gen = NoiseGen::new(96_000.0);
        for i in 0..256 {
            let b = ((i as f64) * 0.13).sin() * 0.5;
            let out = r#gen.tilt(b, 0.0);
            assert!(
                (out - b).abs() < 1e-12,
                "tilt(0.0) diverged: in={b}, out={out}"
            );
        }
    }

    #[test]
    fn signed_pct_formatting_is_fixed_width() {
        for v in [-100, -50, -5, 0, 5, 50, 100] {
            let s = format_signed_pct(v);
            assert_eq!(s.len(), 5, "unexpected width for {v}: {s:?}");
        }
    }

    #[test]
    fn reverb_disabled_outputs_dry_signal() {
        let mut reverb = Reverb::new(48_000.0, 2);
        reverb.begin_frame(false, 50);
        assert_eq!(reverb.process_channel(0, 0.42), 0.42);
        assert_eq!(reverb.process_channel(1, -0.37), -0.37);
    }

    #[test]
    fn reverb_is_bounded_and_stereo_decorrelated() {
        let sr = 48_000.0;
        let mut r#gen = NoiseGen::new(sr * 2.0);
        let mut reverb = Reverb::new(sr, 2);

        let mut dry_sq = 0.0;
        let mut wet_sq = 0.0;
        let mut diff_sq = 0.0;
        let frames = 48_000 * 3; // 3 seconds

        for i in 0..frames {
            reverb.begin_frame(true, 100);
            let d = r#gen.sample(false, 30, 25);
            let l = reverb.process_channel(0, d);
            let r = reverb.process_channel(1, d);
            assert!(l.is_finite() && r.is_finite());
            assert!(
                l.abs() < 2.0 && r.abs() < 2.0,
                "reverb blew up: l={l}, r={r}"
            );
            if i >= 48_000 {
                dry_sq += d * d;
                wet_sq += 0.5 * (l * l + r * r);
                diff_sq += (l - r) * (l - r);
            }
        }

        let dry_rms = (dry_sq / (frames - 48_000) as f64).sqrt();
        let wet_rms = (wet_sq / (frames - 48_000) as f64).sqrt();
        let diff_rms = (diff_sq / (frames - 48_000) as f64).sqrt();
        let ratio = wet_rms / dry_rms;

        assert!(
            (0.4..=1.5).contains(&ratio),
            "unexpected wet/dry RMS ratio: {ratio:.3} (dry={dry_rms:.4}, wet={wet_rms:.4})"
        );
        assert!(
            diff_rms > wet_rms * 0.5,
            "stereo channels should be decorrelated: diff_rms={diff_rms:.4}, wet_rms={wet_rms:.4}"
        );
    }

    /// Strip ANSI CSI (`\x1b[...<letter>`) sequences so we can inspect the
    /// visible characters of a rendered frame.
    fn strip_ansi(s: &str) -> String {
        let mut out = String::new();
        let mut chars = s.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\x1b' && chars.peek() == Some(&'[') {
                chars.next();
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else {
                out.push(ch);
            }
        }
        out
    }

    #[test]
    fn row_cells_fit_exact_inner_width() {
        let rows = [
            Row {
                label: "mode".to_string(),
                value: "Brown (oscillating)".to_string(),
                bold_value: true,
                value_color: Color::Yellow,
                accent: None,
                accent_bold: false,
                accent_color: Color::White,
            },
            Row {
                label: "osc".to_string(),
                value: " 30s  ±25%   -[━━━━━━━━━━●━━━━━━━━━━]+".to_string(),
                bold_value: true,
                value_color: Color::Magenta,
                accent: Some("  -3%".to_string()),
                accent_bold: false,
                accent_color: Color::White,
            },
            Row {
                label: "controls".to_string(),
                value: "↑/↓ vol  ←/→ cycle  +/- range  o osc  r/[/] rev  q quit".to_string(),
                bold_value: false,
                value_color: Color::DarkGrey,
                accent: None,
                accent_bold: false,
                accent_color: Color::White,
            },
        ];
        for width in [4, 12, 30, 50, 72, 80, 140] {
            let inner = width - 2;
            for row in &rows {
                let total: usize = row
                    .cells(inner, 8)
                    .iter()
                    .map(|c| c.text.chars().count())
                    .sum();
                assert_eq!(
                    total, inner,
                    "row {:?} mismatched inner width {inner} at terminal width {width}",
                    row.label
                );
            }
        }
    }

    #[test]
    fn build_frame_aligns_borders_and_avoids_bare_newlines() {
        let state = UiState {
            volume: 20,
            osc_enabled: true,
            osc_secs: 30,
            osc_amount: 25,
            osc_pos: -3,
            reverb_enabled: true,
            reverb_mix: 35,
        };
        for width in [12, 40, 80, 150] {
            let frame = build_frame(width, state).unwrap();
            assert!(
                !frame.contains('\n'),
                "raw-mode frame must not contain bare \\n (width={width})"
            );

            // Split on explicit row positioning `\x1b[row;1H` (1-based).
            let mut lines = Vec::new();
            for row_idx in 1..=7 {
                let marker = format!("\x1b[{row_idx};1H");
                let start = frame
                    .find(&marker)
                    .unwrap_or_else(|| panic!("missing {marker} at width={width}"))
                    + marker.len();
                let next_marker = format!("\x1b[{};1H", row_idx + 1);
                let end = frame.find(&next_marker).unwrap_or(frame.len());
                lines.push(strip_ansi(&frame[start..end]));
            }

            assert_eq!(lines.len(), 7);
            for (i, line) in lines.iter().enumerate() {
                assert_eq!(
                    line.chars().count(),
                    width,
                    "line {i} width mismatch at width={width}: {line:?}"
                );
                let first = line.chars().next().unwrap();
                let last = line.chars().last().unwrap();
                match i {
                    0 => assert_eq!((first, last), ('┌', '┐')),
                    6 => assert_eq!((first, last), ('└', '┘')),
                    _ => assert_eq!((first, last), ('│', '│')),
                }
            }
        }
    }
}
