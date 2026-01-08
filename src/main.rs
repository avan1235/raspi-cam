use color_eyre::eyre::Context;
use futures_util::{SinkExt, StreamExt};
use std::io::{BufRead, BufReader, Read};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
use tokio_tungstenite::{accept_async, tungstenite::Message};
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

const CAPTURE_WIDTH: u32 = 640;
const CAPTURE_HEIGHT: u32 = 480;

const WEBSOCKET_ADDRESS: &str = "0.0.0.0:8080";

/// MJPEG frame markers
const JPEG_SOI: [u8; 2] = [0xFF, 0xD8]; // Start of Image
const JPEG_EOI: [u8; 2] = [0xFF, 0xD9]; // End of Image

/// Shared state for frame data
struct FrameState {
    /// The latest JPEG frame data
    jpeg_data: Option<Vec<u8>>,
    /// Frame sequence number - incremented each time a new frame is captured
    frame_seq: u64,
}

/// Coordinator for frame distribution
#[derive(Clone)]
struct FrameCoordinator {
    /// Current frame state
    state: Arc<Mutex<FrameState>>,
    /// Number of clients currently connected
    connected_clients: Arc<AtomicUsize>,
    /// Watch channel to notify clients when new frame is available
    frame_notify: watch::Sender<u64>,
    /// Receiver for frame notifications
    frame_receiver: watch::Receiver<u64>,
}

impl FrameCoordinator {
    fn new() -> Self {
        let (frame_notify, frame_receiver) = watch::channel(0u64);
        Self {
            state: Arc::new(Mutex::new(FrameState {
                jpeg_data: None,
                frame_seq: 0,
            })),
            connected_clients: Arc::new(AtomicUsize::new(0)),
            frame_notify,
            frame_receiver,
        }
    }

    /// Called by clients to get the latest frame
    async fn get_latest_frame(&self) -> Option<Vec<u8>> {
        // Get current frame sequence
        let current_seq = {
            let state = self.state.lock().unwrap();
            state.frame_seq
        };

        // If we have a frame already, return it immediately
        if current_seq > 0 {
            let state = self.state.lock().unwrap();
            return state.jpeg_data.clone();
        }

        // Otherwise wait for the first frame
        let mut receiver = self.frame_receiver.clone();

        let result = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if receiver.changed().await.is_err() {
                    return None;
                }
                let new_seq = *receiver.borrow();
                if new_seq > 0 {
                    break;
                }
            }

            let state = self.state.lock().unwrap();
            state.jpeg_data.clone()
        })
            .await;

        match result {
            Ok(data) => data,
            Err(_) => {
                tracing::warn!("Timeout waiting for first frame");
                None
            }
        }
    }

    /// Wait for a new frame (used when client wants to wait for fresh data)
    async fn wait_for_new_frame(&self) -> Option<Vec<u8>> {
        let current_seq = {
            let state = self.state.lock().unwrap();
            state.frame_seq
        };

        let mut receiver = self.frame_receiver.clone();

        let result = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if receiver.changed().await.is_err() {
                    return None;
                }
                let new_seq = *receiver.borrow();
                if new_seq > current_seq {
                    break;
                }
            }

            let state = self.state.lock().unwrap();
            state.jpeg_data.clone()
        })
            .await;

        match result {
            Ok(data) => data,
            Err(_) => {
                tracing::warn!("Timeout waiting for new frame");
                None
            }
        }
    }

    fn increment_clients(&self) {
        self.connected_clients.fetch_add(1, Ordering::SeqCst);
    }

    fn decrement_clients(&self) {
        self.connected_clients.fetch_sub(1, Ordering::SeqCst);
    }

    fn has_clients(&self) -> bool {
        self.connected_clients.load(Ordering::SeqCst) > 0
    }

    /// Called by capture thread to publish a new frame
    fn publish_frame(&self, jpeg_data: Vec<u8>) {
        let new_seq = {
            let mut state = self.state.lock().unwrap();
            state.frame_seq += 1;
            state.jpeg_data = Some(jpeg_data);
            state.frame_seq
        };

        // Notify all waiting clients
        let _ = self.frame_notify.send(new_seq);
    }
}

/// Manages the v4l2-ctl process
struct V4l2Process {
    child: Child,
}

impl V4l2Process {
    fn spawn(device: &str, width: u32, height: u32) -> color_eyre::Result<Self> {
        let child = Command::new("v4l2-ctl")
            .args([
                "-d",
                device,
                "--set-fmt-video-out",
                &format!("width={},height={},pixelformat=YUYV", width, height),
                "--set-fmt-video",
                &format!("width={},height={},pixelformat=MJPG", width, height),
                "--stream-mmap",
                "--stream-to=-", // Output to stdout
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to spawn v4l2-ctl process")?;

        tracing::info!("Started v4l2-ctl process with PID {}", child.id());

        Ok(Self { child })
    }

    fn take_stdout(&mut self) -> Option<std::process::ChildStdout> {
        self.child.stdout.take()
    }
}

impl Drop for V4l2Process {
    fn drop(&mut self) {
        tracing::info!("Terminating v4l2-ctl process");
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// MJPEG stream parser that extracts individual JPEG frames
struct MjpegParser<R: Read> {
    reader: BufReader<R>,
    buffer: Vec<u8>,
}

impl<R: Read> MjpegParser<R> {
    fn new(reader: R) -> Self {
        Self {
            reader: BufReader::with_capacity(1024 * 1024, reader), // 1MB buffer
            buffer: Vec::with_capacity(256 * 1024),                // 256KB initial frame buffer
        }
    }

    /// Read the next JPEG frame from the stream
    fn next_frame(&mut self) -> color_eyre::Result<Option<Vec<u8>>> {
        self.buffer.clear();

        // Find SOI marker (start of JPEG)
        if !self.find_marker(&JPEG_SOI)? {
            return Ok(None);
        }

        self.buffer.extend_from_slice(&JPEG_SOI);

        // Read until EOI marker (end of JPEG)
        loop {
            let byte = match self.read_byte()? {
                Some(b) => b,
                None => return Ok(None),
            };

            self.buffer.push(byte);

            // Check for EOI marker
            if self.buffer.len() >= 2 {
                let len = self.buffer.len();
                if self.buffer[len - 2] == JPEG_EOI[0] && self.buffer[len - 1] == JPEG_EOI[1] {
                    // Found complete JPEG frame
                    return Ok(Some(std::mem::take(&mut self.buffer)));
                }
            }

            // Safety limit - JPEG frames shouldn't be larger than 10MB
            if self.buffer.len() > 10 * 1024 * 1024 {
                tracing::warn!("Frame too large, discarding");
                self.buffer.clear();
                return self.next_frame();
            }
        }
    }

    fn find_marker(&mut self, marker: &[u8; 2]) -> color_eyre::Result<bool> {
        let mut prev_byte: Option<u8> = None;

        loop {
            let byte = match self.read_byte()? {
                Some(b) => b,
                None => return Ok(false),
            };

            if let Some(prev) = prev_byte {
                if prev == marker[0] && byte == marker[1] {
                    return Ok(true);
                }
            }

            prev_byte = Some(byte);
        }
    }

    fn read_byte(&mut self) -> color_eyre::Result<Option<u8>> {
        let mut buf = [0u8; 1];
        match self.reader.read(&mut buf)? {
            0 => Ok(None),
            _ => Ok(Some(buf[0])),
        }
    }
}

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    let coordinator = FrameCoordinator::new();

    // Start WebSocket server
    let coordinator_ws = coordinator.clone();
    tokio::spawn(async move {
        if let Err(e) = run_websocket_server(WEBSOCKET_ADDRESS, coordinator_ws).await {
            tracing::error!("WebSocket server error: {}", e);
        }
    });

    // Run camera capture in blocking thread
    let coordinator_capture = coordinator.clone();
    tokio::task::spawn_blocking(move || {
        if let Err(e) = run_camera_capture(coordinator_capture) {
            tracing::error!("Camera capture error: {}", e);
        }
    })
        .await?;

    Ok(())
}

async fn run_websocket_server(addr: &str, coordinator: FrameCoordinator) -> color_eyre::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("WebSocket server listening on: {}", addr);

    while let Ok((stream, addr)) = listener.accept().await {
        tracing::info!("New WebSocket connection from: {}", addr);
        let coordinator = coordinator.clone();
        tokio::spawn(handle_client(stream, coordinator, addr));
    }

    Ok(())
}

async fn handle_client(
    stream: TcpStream,
    coordinator: FrameCoordinator,
    client_addr: std::net::SocketAddr,
) {
    tracing::debug!("Starting WebSocket handshake with client: {}", client_addr);
    let ws_stream = match accept_async(stream).await {
        Ok(ws) => {
            tracing::debug!("WebSocket handshake successful with: {}", client_addr);
            ws
        }
        Err(e) => {
            tracing::error!(
                "Error during WebSocket handshake with {}: {}",
                client_addr,
                e
            );
            return;
        }
    };

    coordinator.increment_clients();
    let (mut write, mut read) = ws_stream.split();

    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                let text = text.trim();
                if text == "s" || text == "snap" {
                    // Get latest frame immediately
                    tracing::debug!("Received snapshot command from {}", client_addr);

                    let jpeg_data = match coordinator.get_latest_frame().await {
                        Some(data) => data,
                        None => {
                            tracing::warn!("No frame available for client {}", client_addr);
                            continue;
                        }
                    };

                    tracing::debug!(
                        "Sending frame to {}: {} bytes",
                        client_addr,
                        jpeg_data.len()
                    );

                    if let Err(e) = write.send(Message::Binary(jpeg_data.into())).await {
                        tracing::error!("Error sending frame to {}: {}", client_addr, e);
                        break;
                    }
                } else if text == "w" || text == "wait" {
                    // Wait for a new frame
                    tracing::debug!("Received wait command from {}", client_addr);

                    let jpeg_data = match coordinator.wait_for_new_frame().await {
                        Some(data) => data,
                        None => {
                            tracing::warn!("Timeout waiting for new frame for client {}", client_addr);
                            continue;
                        }
                    };

                    tracing::debug!(
                        "Sending new frame to {}: {} bytes",
                        client_addr,
                        jpeg_data.len()
                    );

                    if let Err(e) = write.send(Message::Binary(jpeg_data.into())).await {
                        tracing::error!("Error sending frame to {}: {}", client_addr, e);
                        break;
                    }
                }
            }
            Ok(Message::Close(frame)) => {
                tracing::info!("Client {} disconnected: {:?}", client_addr, frame);
                break;
            }
            Ok(Message::Ping(_))
            | Ok(Message::Pong(_))
            | Ok(Message::Binary(_))
            | Ok(Message::Frame(_)) => {}
            Err(e) => {
                tracing::error!("WebSocket error with {}: {}", client_addr, e);
                break;
            }
        }
    }

    coordinator.decrement_clients();
    tracing::debug!("Connection handler for {} terminated", client_addr);
}

fn run_camera_capture(coordinator: FrameCoordinator) -> color_eyre::Result<()> {
    // Find the appropriate video device
    // /dev/video11 is typically the ISP output on Raspberry Pi
    let device = "/dev/video11";

    tracing::info!(
        "Starting v4l2-ctl capture from {} at {}x{}",
        device,
        CAPTURE_WIDTH,
        CAPTURE_HEIGHT
    );

    let mut process = V4l2Process::spawn(device, CAPTURE_WIDTH, CAPTURE_HEIGHT)?;

    let stdout = process
        .take_stdout()
        .ok_or_else(|| color_eyre::eyre::eyre!("Failed to get stdout from v4l2-ctl"))?;

    let mut parser = MjpegParser::new(stdout);
    let mut frame_count = 0u64;
    let mut last_log = std::time::Instant::now();

    tracing::info!("Camera capture loop starting");

    loop {
        match parser.next_frame() {
            Ok(Some(jpeg_data)) => {
                frame_count += 1;

                // Log frame rate periodically
                if last_log.elapsed() >= Duration::from_secs(10) {
                    tracing::info!(
                        "Captured {} frames, latest frame size: {} bytes, clients: {}",
                        frame_count,
                        jpeg_data.len(),
                        coordinator.connected_clients.load(Ordering::SeqCst)
                    );
                    last_log = std::time::Instant::now();
                }

                tracing::trace!("Frame {} captured: {} bytes", frame_count, jpeg_data.len());

                // Publish frame to coordinator
                coordinator.publish_frame(jpeg_data);
            }
            Ok(None) => {
                tracing::warn!("End of stream from v4l2-ctl");
                break;
            }
            Err(e) => {
                tracing::error!("Error reading frame: {}", e);
                // Try to continue on error
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }

    tracing::info!("Camera capture loop ended after {} frames", frame_count);
    Ok(())
}