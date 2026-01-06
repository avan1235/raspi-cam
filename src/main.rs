use crate::buffer::DoubleBuffer;
use color_eyre::eyre::Context;
use futures_util::{SinkExt, StreamExt};
use libcamera::camera::{Camera, CameraConfiguration, CameraConfigurationStatus};
use libcamera::camera_manager::CameraManager;
use libcamera::framebuffer::{AsFrameBuffer, FrameMetadataStatus};
use libcamera::framebuffer_allocator::{FrameBuffer, FrameBufferAllocator};
use libcamera::framebuffer_map::MemoryMappedFrameBuffer;
use libcamera::geometry::Size;
use libcamera::pixel_format::PixelFormat;
use libcamera::request::ReuseFlag;
use libcamera::stream::{StreamConfigurationRef, StreamRole};
use libcamera::*;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use turbojpeg::{Compressor, Image, PixelFormat as TJPixelFormat, Subsamp};

mod buffer;
mod yuyv;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const WEBSOCKET_ADDRESS: &str = "0.0.0.0:8080";
const JPEG_QUALITY: i32 = 85;

/// Tracks the number of connected clients and signals when clients connect/disconnect
#[derive(Clone)]
struct ClientCounter {
    inner: Arc<ClientCounterInner>,
}

struct ClientCounterInner {
    count: AtomicUsize,
    condvar: Condvar,
    mutex: Mutex<()>,
}

impl ClientCounter {
    fn new() -> Self {
        Self {
            inner: Arc::new(ClientCounterInner {
                count: AtomicUsize::new(0),
                condvar: Condvar::new(),
                mutex: Mutex::new(()),
            }),
        }
    }

    fn increment(&self) {
        let old_count = self.inner.count.fetch_add(1, Ordering::SeqCst);
        if old_count == 0 {
            // First client connected, wake up the camera thread
            self.inner.condvar.notify_all();
        }
        tracing::debug!("Client connected, total clients: {}", old_count + 1);
    }

    fn decrement(&self) {
        let old_count = self.inner.count.fetch_sub(1, Ordering::SeqCst);
        tracing::debug!("Client disconnected, total clients: {}", old_count - 1);
    }

    fn has_clients(&self) -> bool {
        self.inner.count.load(Ordering::SeqCst) > 0
    }

    /// Blocks until at least one client is connected
    fn wait_for_clients(&self) {
        let guard = self.inner.mutex.lock().unwrap();
        let _guard = self
            .inner
            .condvar
            .wait_while(guard, |_| !self.has_clients())
            .unwrap();
    }
}

/// RAII guard that decrements the client count when dropped
struct ClientGuard {
    counter: ClientCounter,
}

impl ClientGuard {
    fn new(counter: ClientCounter) -> Self {
        counter.increment();
        Self { counter }
    }
}

impl Drop for ClientGuard {
    fn drop(&mut self) {
        self.counter.decrement();
    }
}

// Shared state for the latest frame
#[derive(Clone)]
struct SharedFrameBuffer {
    data: Arc<Mutex<Option<Vec<u8>>>>,
    width: u32,
    height: u32,
}

impl SharedFrameBuffer {
    fn new(width: u32, height: u32) -> Self {
        Self {
            data: Arc::new(Mutex::new(None)),
            width,
            height,
        }
    }

    fn update(&self, frame_data: &[u8]) {
        let mut data = self.data.lock().unwrap();
        *data = Some(frame_data.to_vec());
    }

    fn get_rgb(&self) -> Option<Vec<u8>> {
        let data = self.data.lock().unwrap();
        data.clone()
    }

    fn clear(&self) {
        let mut data = self.data.lock().unwrap();
        *data = None;
    }
}

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    let frame_buffer = SharedFrameBuffer::new(WIDTH, HEIGHT);
    let client_counter = ClientCounter::new();

    // Start WebSocket server
    let frame_buffer_ws = frame_buffer.clone();
    let client_counter_ws = client_counter.clone();
    tokio::spawn(async move {
        if let Err(e) =
            run_websocket_server(WEBSOCKET_ADDRESS, frame_buffer_ws, client_counter_ws).await
        {
            tracing::error!("WebSocket server error: {}", e);
        }
    });

    let client_counter_cam = client_counter.clone();
    tokio::task::spawn_blocking(move || {
        if let Err(e) = run_camera_capture(frame_buffer, client_counter_cam) {
            tracing::error!("Camera capture error: {}", e);
        }
    })
        .await?;

    Ok(())
}

async fn run_websocket_server(
    addr: &str,
    frame_buffer: SharedFrameBuffer,
    client_counter: ClientCounter,
) -> color_eyre::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("WebSocket server listening on: {}", addr);

    while let Ok((stream, addr)) = listener.accept().await {
        tracing::info!("New WebSocket connection from: {}", addr);
        tracing::debug!(
            "Client address details: {:?}, local address: {:?}",
            addr,
            stream.local_addr()
        );
        let frame_buffer = frame_buffer.clone();
        let client_counter = client_counter.clone();
        tokio::spawn(handle_client(stream, frame_buffer, client_counter, addr));
    }

    Ok(())
}

async fn handle_client(
    stream: TcpStream,
    frame_buffer: SharedFrameBuffer,
    client_counter: ClientCounter,
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

    // Create the guard after successful handshake - this increments the counter
    // and will automatically decrement when the function returns (guard is dropped)
    let _client_guard = ClientGuard::new(client_counter);

    let (mut write, mut read) = ws_stream.split();
    tracing::debug!(
        "WebSocket connection established with {}, waiting for messages",
        client_addr
    );

    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                tracing::debug!("Received text message from {}: {:?}", client_addr, text);
                if text.trim() == "s" {
                    tracing::debug!(
                        "Received 's' command from {}, sending latest frame",
                        client_addr
                    );

                    let rgb_data = match frame_buffer.get_rgb() {
                        Some(data) => data,
                        None => {
                            tracing::warn!("No frame available yet for client {}", client_addr);
                            continue;
                        }
                    };

                    let width = frame_buffer.width;
                    let height = frame_buffer.height;

                    // Move JPEG encoding to blocking thread pool
                    let encode_start = std::time::Instant::now();
                    let jpeg_data = match tokio::task::spawn_blocking(move || {
                        encode_as_jpeg_turbo(&rgb_data, width, height, JPEG_QUALITY)
                    })
                        .await
                    {
                        Ok(Ok(data)) => data,
                        Ok(Err(e)) => {
                            tracing::error!(
                                "JPEG encoding error for client {}: {}",
                                client_addr,
                                e
                            );
                            continue;
                        }
                        Err(e) => {
                            tracing::error!("Task join error for client {}: {}", client_addr, e);
                            continue;
                        }
                    };

                    let encode_duration = encode_start.elapsed();
                    let data_len = jpeg_data.len();
                    tracing::debug!(
                        "JPEG encoding completed for {}: {} bytes in {:?}",
                        client_addr,
                        data_len,
                        encode_duration
                    );

                    let send_start = std::time::Instant::now();
                    if let Err(e) = write.send(Message::Binary(jpeg_data.into())).await {
                        tracing::error!("Error sending frame to {}: {}", client_addr, e);
                        break;
                    }
                    let send_duration = send_start.elapsed();
                    tracing::debug!(
                        "Frame sent to {} successfully: {} bytes in {:?} ({:.2} MB/s)",
                        client_addr,
                        data_len,
                        send_duration,
                        data_len as f64 / send_duration.as_secs_f64() / 1_000_000.0
                    );
                } else {
                    tracing::debug!("Received unknown command from {}: {:?}", client_addr, text);
                }
            }
            Ok(Message::Close(frame)) => {
                tracing::info!("Client {} disconnected: {:?}", client_addr, frame);
                break;
            }
            Ok(Message::Ping(data)) => {
                tracing::debug!("Received ping from {}: {} bytes", client_addr, data.len());
            }
            Ok(Message::Pong(data)) => {
                tracing::debug!("Received pong from {}: {} bytes", client_addr, data.len());
            }
            Ok(Message::Binary(data)) => {
                tracing::debug!(
                    "Received binary message from {}: {} bytes",
                    client_addr,
                    data.len()
                );
            }
            Ok(Message::Frame(_)) => {
                tracing::debug!("Received raw frame from {}", client_addr);
            }
            Err(e) => {
                tracing::error!("WebSocket error with {}: {}", client_addr, e);
                break;
            }
        }
    }

    tracing::debug!("Connection handler for {} terminated", client_addr);
    // _client_guard is dropped here, decrementing the counter
}

fn run_camera_capture(
    frame_buffer: SharedFrameBuffer,
    client_counter: ClientCounter,
) -> color_eyre::Result<()> {
    let camera_manager = CameraManager::new()?;
    let cameras = camera_manager.cameras();

    let cam = cameras.get(0).expect("No cameras found");

    tracing::info!(
        "Using camera: {}",
        *cam.properties().get::<properties::Model>()?
    );

    let mut cam = cam.acquire()?;

    let stream_formats = vec![yuyv::YuyvStream];
    let camera_stream = stream_formats
        .into_iter()
        .find_map(|stream| stream.is_supported(&cam).map(|cfg| (stream, cfg)));

    let Some((camera_stream, mut cfg)) = camera_stream else {
        color_eyre::eyre::bail!("No supported stream format found");
    };

    cfg.get_mut(0).unwrap().set_size(Size::new(WIDTH, HEIGHT));

    match cfg.validate() {
        CameraConfigurationStatus::Adjusted => {
            tracing::warn!(
                "Camera configuration was adjusted after changing frame size: {cfg:#?}"
            )
        }
        CameraConfigurationStatus::Invalid => color_eyre::eyre::bail!(
            "Error validating camera configuration after changing frame_size"
        ),
        _ => {}
    }

    cam.configure(&mut cfg)
        .context("Unable to configure camera")?;

    let mut alloc = FrameBufferAllocator::new(&cam);

    let cfg_ref = cfg.get(0).unwrap();
    let stream = cfg_ref.stream().unwrap();
    let buffers = alloc.alloc(&stream)?;
    tracing::debug!("Allocated {} buffers", buffers.len());

    let buffers = buffers
        .into_iter()
        .map(|buf| MemoryMappedFrameBuffer::new(buf).unwrap())
        .collect::<Vec<_>>();

    // Store requests in a Vec that we can drain and refill
    let mut pending_reqs: Vec<_> = buffers
        .into_iter()
        .enumerate()
        .map(|(i, buf)| {
            let mut req = cam.create_request(Some(i as u64)).unwrap();
            req.add_buffer(&stream, buf).unwrap();
            req
        })
        .collect();

    let (tx, rx) = std::sync::mpsc::channel();
    cam.on_request_completed(move |req| {
        tx.send(req).unwrap();
    });

    let mut buffer = DoubleBuffer::new(cfg_ref.get_size());

    // Main loop: alternate between waiting for clients and capturing
    loop {
        // Wait for at least one client to connect
        tracing::info!("Waiting for clients to connect...");
        client_counter.wait_for_clients();
        tracing::info!("Client connected, starting camera capture");

        // Start camera
        cam.start(None)?;

        // Queue initial requests
        for req in pending_reqs.drain(..) {
            tracing::debug!("Request queued for execution: {req:#?}");
            cam.queue_request(req).map_err(|(_, e)| e)?;
        }

        let mut last_capture = std::time::Instant::now();

        // Capture loop - runs while clients are connected
        loop {
            // Use timeout to periodically check client count
            let recv_result = rx.recv_timeout(Duration::from_millis(500));

            // Check if all clients disconnected
            if !client_counter.has_clients() {
                tracing::info!("All clients disconnected, stopping camera");
                break;
            }

            let mut req = match recv_result {
                Ok(req) => req,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    color_eyre::eyre::bail!("Camera request channel disconnected");
                }
            };

            tracing::debug!("Took {:?} since last capture", last_capture.elapsed());

            let frame_data = {
                let instant = std::time::Instant::now();

                tracing::debug!("Camera request {req:?} completed!");
                tracing::trace!("Metadata: {:#?}", req.metadata());

                let framebuffer: &MemoryMappedFrameBuffer<FrameBuffer> =
                    req.buffer(&stream).unwrap();
                tracing::trace!("FrameBuffer metadata: {:#?}", framebuffer.metadata());
                let frame_metadata_status = framebuffer.metadata().unwrap().status();
                if frame_metadata_status != FrameMetadataStatus::Success {
                    tracing::error!("Frame metadata status: {:?}", frame_metadata_status);
                    req.reuse(ReuseFlag::REUSE_BUFFERS);
                    cam.queue_request(req).map_err(|(_, e)| e)?;
                    continue;
                }
                let bytes_used = framebuffer
                    .metadata()
                    .unwrap()
                    .planes()
                    .get(0)
                    .unwrap()
                    .bytes_used as usize;

                let planes = framebuffer.data();
                tracing::trace!("Data Planes: {:?}", planes.len());
                let frame_data = planes.get(0).unwrap();
                tracing::debug!("Frame captured in {:?}", instant.elapsed());

                &frame_data[..bytes_used]
            };

            let instant = std::time::Instant::now();
            camera_stream.convert_frame(&cfg_ref, frame_data, &mut buffer)?;
            tracing::debug!("Converted in {:?}", instant.elapsed());

            req.reuse(ReuseFlag::REUSE_BUFFERS);
            cam.queue_request(req).map_err(|(_, e)| e)?;

            // Update shared frame buffer
            frame_buffer.update(buffer.deref());

            last_capture = std::time::Instant::now();
            buffer.swap();
        }

        // Stop camera and clear frame buffer
        cam.stop()?;
        frame_buffer.clear();

        // Collect all pending requests back
        while let Ok(req) = rx.try_recv() {
            pending_reqs.push(req);
        }

        // Wait a bit for any in-flight requests to complete
        std::thread::sleep(Duration::from_millis(100));
        while let Ok(req) = rx.try_recv() {
            pending_reqs.push(req);
        }
    }
}

thread_local! {
    static COMPRESSOR: std::cell::RefCell<Option<Compressor>> = const { std::cell::RefCell::new(None) };
}

fn encode_as_jpeg_turbo(
    rgb_data: &[u8],
    width: u32,
    height: u32,
    quality: i32,
) -> Result<Vec<u8>, String> {
    COMPRESSOR.with(|comp_cell| {
        let mut comp_opt = comp_cell.borrow_mut();

        if comp_opt.is_none() {
            let mut compressor = Compressor::new().map_err(|e| e.to_string())?;
            compressor.set_quality(quality).map_err(|e| e.to_string())?;
            compressor
                .set_subsamp(Subsamp::Sub2x2)
                .map_err(|e| e.to_string())?;
            *comp_opt = Some(compressor);
        }

        let compressor = comp_opt.as_mut().unwrap();

        let image = Image {
            pixels: rgb_data,
            width: width as usize,
            height: height as usize,
            pitch: (width * 3) as usize,
            format: TJPixelFormat::RGB,
        };

        compressor.compress_to_vec(image).map_err(|e| e.to_string())
    })
}

trait CameraStream {
    fn name(&self) -> &'static str;
    fn is_supported(&self, camera: &Camera) -> Option<CameraConfiguration>;
    fn convert_frame(
        &self,
        configuration: &StreamConfigurationRef,
        data: &[u8],
        target_buffer: &mut [u8],
    ) -> color_eyre::Result<()>;
}

fn supports_configuration(cam: &Camera, format: PixelFormat) -> Option<CameraConfiguration> {
    let mut cfgs = cam.generate_configuration(&[StreamRole::VideoRecording])?;
    cfgs.get_mut(0)?.set_pixel_format(format);

    tracing::trace!("Generated config: {cfgs:#?}");

    match cfgs.validate() {
        CameraConfigurationStatus::Valid => tracing::debug!("Camera configuration {format} valid!"),
        CameraConfigurationStatus::Adjusted => {
            tracing::trace!("Camera configuration was adjusted: {cfgs:#?}")
        }
        CameraConfigurationStatus::Invalid => {
            tracing::trace!("Error validating camera configuration for {format}")
        }
    }

    if cfgs.get(0).unwrap().get_pixel_format() != format {
        return None;
    }

    Some(cfgs)
}