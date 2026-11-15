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
/// * `q` or `Esc`     — quit.
/// * `Ctrl+C`         — quit.
use std::io::stdout;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI8, AtomicU8, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream};
use crossterm::Command;
use crossterm::cursor::MoveTo;
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

/// Build a playing output `Stream` for the requested device.
fn create_stream(
    device: &Device,
    volume: Arc<AtomicU8>,
    osc_enabled: Arc<AtomicBool>,
    osc_secs: Arc<AtomicUsize>,
    osc_amount: Arc<AtomicU8>,
    osc_pos: Arc<AtomicI8>,
) -> Result<Stream, String> {
    let config = device
        .default_output_config()
        .map_err(|e| format!("no output config: {e}"))?;
    let samples_per_sec = config.sample_rate().0 as f64 * config.channels() as f64;

    let stream = match config.sample_format() {
        SampleFormat::F32 => {
            let mut r#gen = NoiseGen::new(samples_per_sec);
            device
                .build_output_stream(
                    &config.config(),
                    move |buffer: &mut [f32], _| {
                        let gain = volume.load(Ordering::Relaxed) as f32 / 100.0;
                        let enabled = osc_enabled.load(Ordering::Relaxed);
                        let period = osc_secs.load(Ordering::Relaxed);
                        let amount = osc_amount.load(Ordering::Relaxed);
                        for s in buffer.iter_mut() {
                            *s = (r#gen.sample(enabled, period, amount) as f32 * gain)
                                .clamp(-1.0, 1.0);
                        }
                        let pos_pct = (r#gen.current_offset() * 100.0).round() as i8;
                        osc_pos.store(pos_pct, Ordering::Relaxed);
                    },
                    |err| eprintln!("audio stream error: {err}"),
                    None,
                )
                .map_err(|e| e.to_string())?
        }
        SampleFormat::I16 => {
            let mut r#gen = NoiseGen::new(samples_per_sec);
            device
                .build_output_stream(
                    &config.config(),
                    move |buffer: &mut [i16], _| {
                        let gain = volume.load(Ordering::Relaxed) as f64 / 100.0;
                        let enabled = osc_enabled.load(Ordering::Relaxed);
                        let period = osc_secs.load(Ordering::Relaxed);
                        let amount = osc_amount.load(Ordering::Relaxed);
                        for s in buffer.iter_mut() {
                            let v = r#gen.sample(enabled, period, amount) * gain * 32_767.0;
                            *s = v.clamp(-32_768.0, 32_767.0) as i16;
                        }
                        let pos_pct = (r#gen.current_offset() * 100.0).round() as i8;
                        osc_pos.store(pos_pct, Ordering::Relaxed);
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
    /// Render the row as `[left-pad][label][gap][value][gap][right-pad][accent]`.
    ///
    /// `inner_width` is the box interior (terminal width minus the 4 border
    /// columns). The right padding is computed so the accent sits flush against
    /// the right border. If the content is wider than the terminal (a very
    /// narrow viewport) the padding clamps to zero so we degrade gracefully
    /// instead of overflowing the box.
    fn cells(&self, inner_width: usize) -> Vec<Cell> {
        let label_w = self.label.chars().count();
        let value_w = self.value.chars().count();
        let accent_w = self.accent.as_ref().map_or(0, |a| a.chars().count());

        let left_pad = "  ";
        let gap1 = "  ";
        let gap2 = "  ";

        let fixed = left_pad.chars().count()
            + label_w
            + gap1.chars().count()
            + value_w
            + gap2.chars().count()
            + accent_w;

        let right_pad = inner_width.saturating_sub(fixed);

        let mut cells = vec![
            Cell {
                text: left_pad.to_string(),
                bold: false,
                color: Color::Grey,
            },
            Cell {
                text: self.label.clone(),
                bold: true,
                color: Color::White,
            },
            Cell {
                text: gap1.to_string(),
                bold: false,
                color: self.value_color,
            },
            Cell {
                text: self.value.clone(),
                bold: self.bold_value,
                color: self.value_color,
            },
            Cell {
                text: gap2.to_string(),
                bold: false,
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
                text: accent.clone(),
                bold: self.accent_bold,
                color: self.accent_color,
            });
        }
        cells
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

/// Paint the status panel describing the current state.
fn render(
    volume: u8,
    osc_enabled: bool,
    osc_secs: usize,
    osc_amount: u8,
    osc_pos: i8,
) -> std::io::Result<()> {
    // Resolve the actual terminal width so the box always fits the real
    // viewport instead of a hard-coded ~65-column frame that overflows narrow
    // terminals or looks sparse on wide ones.
    let width = terminal::size().map(|(w, _)| w as usize).unwrap_or(80);
    let inner = width.saturating_sub(4); // interior = width - ("│" + "│" + two side spaces)
    let border_inner = width.saturating_sub(2); // dash run between ┌ and ┐

    let filled = (volume as usize) / 5; // 0..=20 segments
    let mut bar = String::with_capacity(22);
    bar.push('│');
    for i in 0..20 {
        bar.push(if i < filled { '█' } else { '░' });
    }
    bar.push('│');

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
        value: "↑/↓ vol  ←/→ cycle  +/- range  o osc  q quit".to_string(),
        bold_value: false,
        value_color: Color::DarkGrey,
        accent: None,
        accent_bold: false,
        accent_color: Color::White,
    };
    let rows = [mode, osc, volume_row, controls];

    // Top border with a centered title and a bottom border, both sized to the
    // live width (no hand-counted dashes).
    let title = " noise ";
    let title_w = title.chars().count();
    let left_pad = if border_inner > title_w {
        (border_inner - title_w) / 2
    } else {
        0
    };
    let right_pad = if border_inner > title_w {
        border_inner - title_w - left_pad
    } else {
        0
    };
    let top = format!(
        "┌{}{}{}┐",
        "─".repeat(left_pad),
        title,
        "─".repeat(right_pad)
    );
    let bottom = format!("└{}┘", "─".repeat(border_inner));

    // Serialize every command (styling + text) into a single ANSI string and
    // write it in one shot.
    let mut out = String::with_capacity(1024);
    push_command(&mut out, MoveTo(0, 0))?;
    push_command(&mut out, SetStyle(ContentStyle::default()))?;
    push_command(&mut out, SetAttribute(Attribute::Bold))?;
    push_command(&mut out, SetForegroundColor(Color::White))?;
    push_command(&mut out, Print(top))?;
    push_command(&mut out, SetAttribute(Attribute::Reset))?;

    for row in &rows {
        for c in row.cells(inner) {
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
        push_command(&mut out, SetStyle(ContentStyle::default()))?;
        push_command(&mut out, SetForegroundColor(Color::Grey))?;
        push_command(&mut out, Print("│"))?;
        push_command(&mut out, Print("\n"))?;
    }

    push_command(&mut out, SetAttribute(Attribute::Bold))?;
    push_command(&mut out, SetForegroundColor(Color::White))?;
    push_command(&mut out, Print(bottom))?;
    push_command(&mut out, SetStyle(ContentStyle::default()))?;
    push_command(&mut out, Print("\n"))?;

    std::io::Write::write_all(&mut stdout(), out.as_bytes())?;
    Ok(())
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
    let volume = Arc::new(AtomicU8::new(20));
    let osc_enabled = Arc::new(AtomicBool::new(true));
    let osc_secs = Arc::new(AtomicUsize::new(30)); // 30s full oscillation cycle
    let osc_amount = Arc::new(AtomicU8::new(25)); // ±25% away from Brown
    let osc_pos = Arc::new(AtomicI8::new(0)); // current position in -100..=100%

    // Start audio.
    let host = cpal::default_host();
    let device = match host.default_output_device() {
        Some(d) => d,
        None => {
            eprintln!("error: no audio output device found");
            std::process::exit(1);
        }
    };
    let stream = match create_stream(
        &device,
        volume.clone(),
        osc_enabled.clone(),
        osc_secs.clone(),
        osc_amount.clone(),
        osc_pos.clone(),
    ) {
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
    let _ = execute!(stdout(), Clear(ClearType::All));

    // Initial paint.
    let _ = render(
        volume.load(Ordering::Relaxed),
        osc_enabled.load(Ordering::Relaxed),
        osc_secs.load(Ordering::Relaxed),
        osc_amount.load(Ordering::Relaxed),
        osc_pos.load(Ordering::Relaxed),
    );

    // Event loop: poll with a short timeout so we can also react to the
    // signal handler thread and animate the oscillation slider.
    let mut last_volume = volume.load(Ordering::Relaxed);
    let mut last_enabled = osc_enabled.load(Ordering::Relaxed);
    let mut last_osc_secs = osc_secs.load(Ordering::Relaxed);
    let mut last_osc_amount = osc_amount.load(Ordering::Relaxed);
    let mut last_osc_pos = osc_pos.load(Ordering::Relaxed);
    loop {
        if !running.load(Ordering::Relaxed) {
            break;
        }

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
                            let v = volume.load(Ordering::Relaxed);
                            if v < 100 {
                                volume.store(v.saturating_add(5).min(100), Ordering::Relaxed);
                            }
                        }
                        KeyCode::Down => {
                            let v = volume.load(Ordering::Relaxed);
                            if v > 0 {
                                volume.store(v.saturating_sub(5), Ordering::Relaxed);
                            }
                        }
                        KeyCode::Right => {
                            let s = osc_secs.load(Ordering::Relaxed);
                            osc_secs.store((s + 5).min(300), Ordering::Relaxed);
                        }
                        KeyCode::Left => {
                            let s = osc_secs.load(Ordering::Relaxed);
                            osc_secs.store(s.saturating_sub(5).max(5), Ordering::Relaxed);
                        }
                        KeyCode::Char('+') | KeyCode::Char('=') => {
                            let a = osc_amount.load(Ordering::Relaxed);
                            osc_amount.store(a.saturating_add(5).min(100), Ordering::Relaxed);
                        }
                        KeyCode::Char('-') | KeyCode::Char('_') => {
                            let a = osc_amount.load(Ordering::Relaxed);
                            osc_amount.store(a.saturating_sub(5).max(5), Ordering::Relaxed);
                        }
                        KeyCode::Char('o') | KeyCode::Char('O') | KeyCode::Char(' ') => {
                            let next = !osc_enabled.load(Ordering::Relaxed);
                            osc_enabled.store(next, Ordering::Relaxed);
                        }
                        KeyCode::Char('q') | KeyCode::Esc => {
                            running.store(false, Ordering::SeqCst);
                            break;
                        }
                        _ => {}
                    }
                }
                // Ignore non-key events and key repeats/releases.
                _ => {}
            },
            Ok(false) => {}
            Err(e) => {
                eprintln!("input error: {e}");
                break;
            }
        }

        // Redraw only when something changed, to avoid flicker.
        let cur_volume = volume.load(Ordering::Relaxed);
        let cur_enabled = osc_enabled.load(Ordering::Relaxed);
        let cur_osc_secs = osc_secs.load(Ordering::Relaxed);
        let cur_osc_amount = osc_amount.load(Ordering::Relaxed);
        let cur_osc_pos = osc_pos.load(Ordering::Relaxed);
        let pos_changed = cur_enabled && cur_osc_pos != last_osc_pos;
        if cur_volume != last_volume
            || cur_enabled != last_enabled
            || cur_osc_secs != last_osc_secs
            || cur_osc_amount != last_osc_amount
            || pos_changed
        {
            last_volume = cur_volume;
            last_enabled = cur_enabled;
            last_osc_secs = cur_osc_secs;
            last_osc_amount = cur_osc_amount;
            last_osc_pos = cur_osc_pos;
            let _ = render(
                cur_volume,
                cur_enabled,
                cur_osc_secs,
                cur_osc_amount,
                cur_osc_pos,
            );
        }
    }

    // Clean shutdown.
    let _ = terminal::disable_raw_mode();
    // Dropping the stream stops playback.
    drop(stream);

    // Wipe the status panel so nothing is left on screen.
    let _ = execute!(stdout(), Clear(ClearType::All));
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
}
