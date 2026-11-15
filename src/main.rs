/// # noise
///
/// A continuous white / pink / brown / oscillating pink↔brown noise generator
/// that plays to your default audio output device until you press `Ctrl+C`.
///
/// Controls:
/// * `Up` / `Down`    — increase / decrease volume (5% steps).
/// * `Left` / `Right` — decrease / increase Pink↔Brown cycle length (5s steps).
/// * `n`              — cycle the noise type (White → Pink → Brown → Pink↔Brown).
/// * `q` or `Esc`     — quit.
/// * `Ctrl+C`         — quit.
use std::io::stdout;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream};
use crossterm::cursor::MoveTo;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::style::{
    Attribute, Color, ContentStyle, Print, SetAttribute, SetForegroundColor, SetStyle,
};
use crossterm::terminal::{self, Clear, ClearType};

/// A small, allocation-free noise generator.
///
/// All generators share a single `xorshift64` white-noise source. The pink
/// and brown generators keep their own (stateful) filter state.
struct NoiseGen {
    /// `xorshift64` PRNG state (must never be zero).
    state: u64,
    /// The noise type the state currently reflects, so we can reset the
    /// stateful filters cleanly when the user switches type.
    current: usize,
    /// Paul Kellet pink-noise filter states.
    p0: f64,
    p1: f64,
    p2: f64,
    p3: f64,
    p4: f64,
    p5: f64,
    p6: f64,
    /// Brown-noise (AR(1)) integrator state.
    brown: f64,
    /// LFO phase in `[0.0, 1.0)` for Pink↔Brown oscillation.
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
        NoiseGen {
            state: seed,
            current: usize::MAX, // force a reset on the first sample
            p0: 0.0,
            p1: 0.0,
            p2: 0.0,
            p3: 0.0,
            p4: 0.0,
            p5: 0.0,
            p6: 0.0,
            brown: 0.0,
            phase: 0.0,
            samples_per_sec: samples_per_sec.max(1.0),
        }
    }

    /// Zero the stateful filters (called whenever the noise type changes so
    /// switching types does not produce a DC "thump").
    fn reset(&mut self) {
        self.p0 = 0.0;
        self.p1 = 0.0;
        self.p2 = 0.0;
        self.p3 = 0.0;
        self.p4 = 0.0;
        self.p5 = 0.0;
        self.p6 = 0.0;
        self.brown = 0.0;
    }

    /// One sample of uniform white noise in roughly `[-1, 1]`.
    fn white(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        // Map the top 53 bits onto [-1, 1].
        ((x >> 11) as f64) / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    /// Paul Kellet (approximate) pink-noise filter.
    fn pink(&mut self, w: f64) -> f64 {
        self.p0 = 0.99886 * self.p0 + w * 0.0555179;
        self.p1 = 0.99332 * self.p1 + w * 0.0750759;
        self.p2 = 0.96900 * self.p2 + w * 0.1538520;
        self.p3 = 0.86650 * self.p3 + w * 0.3104856;
        self.p4 = 0.55000 * self.p4 + w * 0.5329522;
        self.p5 = -0.7616 * self.p5 - w * 0.0168980;
        let out = self.p0 + self.p1 + self.p2 + self.p3 + self.p4 + self.p5 + self.p6 + 0.5362 * w;
        self.p6 = out * 0.115926;
        out * 0.11
    }

    /// Brownian (red) noise: an integrated random walk.
    ///
    /// The raw AR(1) output is small, so it is gain-normalised to be
    /// roughly comparable in loudness to the other colours.
    fn brown(&mut self, w: f64) -> f64 {
        self.brown = (self.brown + 0.02 * w) / 1.02;
        self.brown * 3.5
    }

    /// Cosine ease in `[0.0, 1.0]` derived from the current LFO phase
    /// (`0.0` = Pink, `1.0` = Brown).
    fn current_mix(&self) -> f64 {
        0.5 * (1.0 - (self.phase * std::f64::consts::TAU).cos())
    }

    /// Produce one output sample (in ~`[-1, 1]`) for the requested type.
    fn sample(&mut self, noise_type: usize, period_secs: usize) -> f64 {
        if noise_type != self.current {
            self.reset();
            self.phase = 0.0;
            self.current = noise_type;
        }
        match noise_type {
            0 => self.white(),
            1 => {
                let w = self.white();
                self.pink(w)
            }
            2 => {
                let w = self.white();
                self.brown(w)
            }
            3 => {
                let period_samples = (period_secs.max(1) as f64) * self.samples_per_sec;
                self.phase = (self.phase + 1.0 / period_samples) % 1.0;
                let mix = self.current_mix();

                // Separate white sources + equal-power crossfade avoids a mid-slide loudness dip.
                let w_pink = self.white();
                let w_brown = self.white();
                let p = self.pink(w_pink);
                let b = self.brown(w_brown);
                p * (1.0 - mix).sqrt() + b * mix.sqrt()
            }
            _ => self.white(),
        }
    }
}

/// Name a noise type by its index.
fn noise_name(noise_type: usize) -> &'static str {
    match noise_type {
        0 => "White",
        1 => "Pink",
        2 => "Brown",
        3 => "Pink↔Brown",
        _ => "White",
    }
}

/// Build a playing output `Stream` for the requested device.
fn create_stream(
    device: &Device,
    volume: Arc<AtomicU8>,
    noise_type: Arc<AtomicUsize>,
    osc_secs: Arc<AtomicUsize>,
    osc_mix: Arc<AtomicU8>,
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
                        let t = noise_type.load(Ordering::Relaxed);
                        let period = osc_secs.load(Ordering::Relaxed);
                        for s in buffer.iter_mut() {
                            *s = (r#gen.sample(t, period) as f32 * gain).clamp(-1.0, 1.0);
                        }
                        if t == 3 {
                            let mix_pct = (r#gen.current_mix() * 100.0).round() as u8;
                            osc_mix.store(mix_pct, Ordering::Relaxed);
                        }
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
                        let t = noise_type.load(Ordering::Relaxed);
                        let period = osc_secs.load(Ordering::Relaxed);
                        for s in buffer.iter_mut() {
                            let v = r#gen.sample(t, period) * gain * 32_767.0;
                            *s = v.clamp(-32_768.0, 32_767.0) as i16;
                        }
                        if t == 3 {
                            let mix_pct = (r#gen.current_mix() * 100.0).round() as u8;
                            osc_mix.store(mix_pct, Ordering::Relaxed);
                        }
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

/// Paint the little status panel describing the current state.
fn render(volume: u8, noise_type: usize, osc_secs: usize, osc_mix: u8) -> std::io::Result<()> {
    let filled = (volume as usize) / 5; // 0..=20 segments
    let mut bar = String::with_capacity(22);
    bar.push('│');
    for i in 0..20 {
        bar.push(if i < filled { '█' } else { '░' });
    }
    bar.push('│');

    let slider_width = 15usize;
    let pos = ((osc_mix.min(100) as usize) * (slider_width - 1) + 50) / 100;
    let mut slider = String::with_capacity(slider_width + 2);
    slider.push('[');
    for i in 0..slider_width {
        if noise_type == 3 && i == pos {
            slider.push('●');
        } else {
            slider.push('─');
        }
    }
    slider.push(']');

    let cycle_color = if noise_type == 3 {
        Color::Magenta
    } else {
        Color::DarkGrey
    };

    execute!(
        stdout(),
        MoveTo(0, 0),
        SetStyle(ContentStyle::default()),
        SetForegroundColor(Color::White),
        SetAttribute(Attribute::Bold),
        Print("   ┌───────────────────────── noise ──────────────────────────┐\n"),
        SetAttribute(Attribute::Reset),
        SetForegroundColor(Color::Grey),
        Print("   │  type    : "),
        SetForegroundColor(Color::Yellow),
        Print(format!("{:<12}", noise_name(noise_type))),
        SetForegroundColor(Color::Grey),
        Print("                                  │\n   │  cycle   : "),
        SetForegroundColor(cycle_color),
        Print(format!("{:>3}s  Pink {slider} Brown", osc_secs)),
        SetForegroundColor(Color::Grey),
        Print("            │\n   │  volume  : "),
        SetForegroundColor(Color::Cyan),
        Print(format!("{bar} {:>3}%", volume)),
        SetForegroundColor(Color::Grey),
        Print(
            "                   │\n   │                                                          │\n"
        ),
        SetForegroundColor(Color::DarkGrey),
        Print("   │  controls: ↑/↓ vol   ←/→ cycle   n mode   q/Ctrl+C quit  │\n"),
        SetStyle(ContentStyle::default()),
        Print("   └──────────────────────────────────────────────────────────┘\n"),
    )
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
    let noise_type = Arc::new(AtomicUsize::new(1)); // 1 = Pink
    let osc_secs = Arc::new(AtomicUsize::new(30)); // 30s Pink↔Brown cycle
    let osc_mix = Arc::new(AtomicU8::new(0)); // 0..=100% Pink→Brown blend

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
        noise_type.clone(),
        osc_secs.clone(),
        osc_mix.clone(),
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
        noise_type.load(Ordering::Relaxed),
        osc_secs.load(Ordering::Relaxed),
        osc_mix.load(Ordering::Relaxed),
    );

    // Event loop: poll with a short timeout so we can also react to the
    // signal handler thread and animate the oscillation slider.
    let mut last_volume = volume.load(Ordering::Relaxed);
    let mut last_type = noise_type.load(Ordering::Relaxed);
    let mut last_osc_secs = osc_secs.load(Ordering::Relaxed);
    let mut last_osc_mix = osc_mix.load(Ordering::Relaxed);
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
                                volume.store(v.saturating_add(5), Ordering::Relaxed);
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
                        KeyCode::Char('n') | KeyCode::Char('N') => {
                            let next = (noise_type.load(Ordering::Relaxed) + 1) % 4;
                            if next == 3 {
                                osc_mix.store(0, Ordering::Relaxed);
                            }
                            noise_type.store(next, Ordering::Relaxed);
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
        let cur_type = noise_type.load(Ordering::Relaxed);
        let cur_osc_secs = osc_secs.load(Ordering::Relaxed);
        let cur_osc_mix = osc_mix.load(Ordering::Relaxed);
        let mix_changed = cur_type == 3 && cur_osc_mix != last_osc_mix;
        if cur_volume != last_volume
            || cur_type != last_type
            || cur_osc_secs != last_osc_secs
            || mix_changed
        {
            last_volume = cur_volume;
            last_type = cur_type;
            last_osc_secs = cur_osc_secs;
            last_osc_mix = cur_osc_mix;
            let _ = render(cur_volume, cur_type, cur_osc_secs, cur_osc_mix);
        }
    }

    // Clean shutdown.
    let _ = terminal::disable_raw_mode();
    // Dropping the stream stops playback.
    drop(stream);

    // Wipe the status panel so nothing is left on screen.
    let _ = execute!(stdout(), Clear(ClearType::All));
}
