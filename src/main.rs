use color_eyre::eyre::Context;
use futures_util::{SinkExt, StreamExt};
use libcamera::camera::{Camera, CameraConfiguration, CameraConfigurationStatus};
use libcamera::camera_manager::CameraManager;
use libcamera::framebuffer::AsFrameBuffer;
use libcamera::framebuffer::FrameMetadataStatus;
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
use tokio::sync::{broadcast, watch};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use turbojpeg::{Compressor, Image, PixelFormat as TJPixelFormat, Subsamp};

use crate::yuyv::YuyvStream;

mod yuyv;

const CAPTURE_WIDTH: u32 = 1920;
const CAPTURE_HEIGHT: u32 = 1080;

const OUTPUT_WIDTH: u32 = 640;
const OUTPUT_HEIGHT: u32 = 360;

const WEBSOCKET_ADDRESS: &str = "0.0.0.0:8080";

// Reduced quality for Pi 3B - balances quality vs encoding speed
const JPEG_QUALITY: i32 = 75;

/// Shared state for frame data and demand signaling
struct FrameState {
    /// The latest RGB frame data (scaled)
    rgb_data: Option<Vec<u8>>,
    /// Frame sequence number - incremented each time a new frame is captured
    frame_seq: u64,
}

/// Coordinator for on-demand frame capture
#[derive(Clone)]
struct FrameCoordinator {
    /// Current frame state
    state: Arc<Mutex<FrameState>>,
    /// Condition variable to wake up camera thread when frames are needed
    frame_needed: Arc<Condvar>,
    /// Number of clients currently waiting for a frame
    waiting_clients: Arc<AtomicUsize>,
    /// Watch channel to notify clients when new frame is available
    frame_notify: watch::Sender<u64>,
    /// Receiver for frame notifications
    frame_receiver: watch::Receiver<u64>,
    /// Frame dimensions
    width: u32,
    height: u32,
}

impl FrameCoordinator {
    fn new(width: u32, height: u32) -> Self {
        let (frame_notify, frame_receiver) = watch::channel(0u64);
        Self {
            state: Arc::new(Mutex::new(FrameState {
                rgb_data: None,
                frame_seq: 0,
            })),
            frame_needed: Arc::new(Condvar::new()),
            waiting_clients: Arc::new(AtomicUsize::new(0)),
            frame_notify,
            frame_receiver,
            width,
            height,
        }
    }

    /// Called by clients to request a frame. Blocks until a fresh frame is available.
    async fn request_frame(&self) -> Option<Vec<u8>> {
        // Get current frame sequence before signaling
        let current_seq = {
            let state = self.state.lock().unwrap();
            state.frame_seq
        };

        // Increment waiting count and signal camera thread
        self.waiting_clients.fetch_add(1, Ordering::SeqCst);
        self.frame_needed.notify_one();

        // Wait for a new frame (sequence number greater than what we saw)
        let mut receiver = self.frame_receiver.clone();

        // Use tokio::select with timeout to avoid indefinite waiting
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

            // Get the frame data
            let state = self.state.lock().unwrap();
            state.rgb_data.clone()
        })
        .await;

        // Decrement waiting count
        self.waiting_clients.fetch_sub(1, Ordering::SeqCst);

        result.unwrap_or_else(|_| {
            tracing::warn!("Timeout waiting for frame");
            None
        })
    }

    /// Called by camera thread to check if any clients need frames
    fn has_waiting_clients(&self) -> bool {
        self.waiting_clients.load(Ordering::SeqCst) > 0
    }

    /// Called by camera thread to wait until clients need frames
    fn wait_for_demand(&self, timeout: Duration) -> bool {
        let state = self.state.lock().unwrap();
        let result = self
            .frame_needed
            .wait_timeout_while(state, timeout, |_| !self.has_waiting_clients());

        match result {
            Ok((_, timeout_result)) => !timeout_result.timed_out(),
            Err(_) => false,
        }
    }

    fn publish_frame(&self, rgb_data: &[u8]) {
        let new_seq = {
            let mut state = self.state.lock().unwrap();
            state.frame_seq += 1;
            state.rgb_data = Some(rgb_data.to_vec());
            state.frame_seq
        };
        let _ = self.frame_notify.send(new_seq);
    }
}

fn scale_rgb_bilinear(
    src: &[u8],
    dst: &mut [u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) {
    let x_ratio = ((src_width - 1) << 16) / dst_width;
    let y_ratio = ((src_height - 1) << 16) / dst_height;

    let src_stride = (src_width * 3) as usize;
    let dst_stride = (dst_width * 3) as usize;

    for dst_y in 0..dst_height {
        let y_fixed = (dst_y * y_ratio) as usize;
        let y_int = y_fixed >> 16;
        let y_frac = (y_fixed & 0xFFFF) as u32;
        let y_inv = 0x10000 - y_frac;

        let src_row0 = y_int * src_stride;
        let src_row1 = (y_int + 1).min((src_height - 1) as usize) * src_stride;
        let dst_row = (dst_y as usize) * dst_stride;

        for dst_x in 0..dst_width {
            let x_fixed = (dst_x * x_ratio) as usize;
            let x_int = x_fixed >> 16;
            let x_frac = (x_fixed & 0xFFFF) as u32;
            let x_inv = 0x10000 - x_frac;

            let x0 = x_int * 3;
            let x1 = ((x_int + 1).min((src_width - 1) as usize)) * 3;

            // Bilinear interpolation for each channel
            for c in 0..3 {
                let p00 = src[src_row0 + x0 + c] as u32;
                let p10 = src[src_row0 + x1 + c] as u32;
                let p01 = src[src_row1 + x0 + c] as u32;
                let p11 = src[src_row1 + x1 + c] as u32;

                // Fixed-point bilinear interpolation
                let top = (p00 * x_inv + p10 * x_frac) >> 16;
                let bottom = (p01 * x_inv + p11 * x_frac) >> 16;
                let value = (top * y_inv + bottom * y_frac) >> 16;

                dst[dst_row + (dst_x as usize) * 3 + c] = value as u8;
            }
        }
    }
}

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    // Output buffer uses scaled dimensions
    let coordinator = FrameCoordinator::new(OUTPUT_WIDTH, OUTPUT_HEIGHT);

    // Start WebSocket server
    let coordinator_ws = coordinator.clone();
    tokio::spawn(async move {
        if let Err(e) = run_websocket_server(WEBSOCKET_ADDRESS, coordinator_ws).await {
            tracing::error!("WebSocket server error: {}", e);
        }
    });

    tokio::task::spawn_blocking(move || {
        if let Err(e) = run_camera_capture(coordinator) {
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

    let (mut write, mut read) = ws_stream.split();

    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                if text.trim() == "s" {
                    tracing::debug!("Received 's' command from {}", client_addr);

                    // Request a fresh frame from the camera
                    let request_start = std::time::Instant::now();
                    let rgb_data = match coordinator.request_frame().await {
                        Some(data) => data,
                        None => {
                            tracing::warn!("No frame available for client {}", client_addr);
                            continue;
                        }
                    };
                    tracing::debug!(
                        "Frame request fulfilled for {} in {:?}",
                        client_addr,
                        request_start.elapsed()
                    );

                    let width = coordinator.width;
                    let height = coordinator.height;

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
                    tracing::debug!(
                        "JPEG encoding completed for {}: {} bytes in {:?}",
                        client_addr,
                        jpeg_data.len(),
                        encode_duration
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

    tracing::debug!("Connection handler for {} terminated", client_addr);
}

fn run_camera_capture(coordinator: FrameCoordinator) -> color_eyre::Result<()> {
    let camera_manager = CameraManager::new()?;
    let cameras = camera_manager.cameras();

    let cam = cameras.get(0).expect("No cameras found");

    tracing::info!(
        "Using camera: {}",
        *cam.properties().get::<properties::Model>()?
    );

    let mut cam = cam.acquire()?;

    let camera_stream = YuyvStream;
    let (camera_stream, mut cfg) = if let Some(cfg) = camera_stream.is_supported(&cam) {
        (camera_stream, cfg)
    } else {
        color_eyre::eyre::bail!("No supported stream format found");
    };

    cfg.get_mut(0)
        .unwrap()
        .set_size(Size::new(CAPTURE_WIDTH, CAPTURE_HEIGHT));

    match cfg.validate() {
        CameraConfigurationStatus::Adjusted => {
            tracing::warn!("Camera configuration was adjusted: {cfg:#?}")
        }
        CameraConfigurationStatus::Invalid => {
            color_eyre::eyre::bail!("Error validating camera configuration")
        }
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

    let reqs = buffers
        .into_iter()
        .enumerate()
        .map(|(i, buf)| {
            let mut req = cam.create_request(Some(i as u64)).unwrap();
            req.add_buffer(&stream, buf).unwrap();
            req
        })
        .collect::<Vec<_>>();

    let (tx, rx) = std::sync::mpsc::channel();
    cam.on_request_completed(move |req| {
        tx.send(req).unwrap();
    });

    cam.start(None)?;

    let mut buffer = vec![0; (CAPTURE_WIDTH * CAPTURE_HEIGHT * 4) as usize];
    let mut scaled = vec![0u8; (OUTPUT_WIDTH * OUTPUT_HEIGHT * 3) as usize];
    let mut last_capture = std::time::Instant::now();

    // Track whether we have requests queued
    let mut requests_queued = 0;

    // Store requests for later queuing
    let mut pending_requests: Vec<_> = reqs.into_iter().collect();

    tracing::info!("Camera capture loop starting - will capture on demand only");

    loop {
        // Wait for client demand if no frames are being processed
        if requests_queued == 0 {
            tracing::debug!("Waiting for client demand...");

            // Wait up to 1 second for demand, then check again
            // This allows the loop to remain responsive
            if !coordinator.wait_for_demand(Duration::from_secs(1)) {
                continue;
            }

            tracing::debug!("Client demand detected, queuing capture requests");

            // Queue requests to start capturing
            while let Some(req) = pending_requests.pop() {
                tracing::debug!("Request queued for execution: {req:#?}");
                cam.queue_request(req).map_err(|(_, e)| e)?;
                requests_queued += 1;
            }
        }

        // Wait for a completed request
        let mut req = match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(req) => {
                requests_queued -= 1;
                req
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                tracing::warn!("Timeout waiting for camera frame");
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                tracing::error!("Camera request channel disconnected");
                break;
            }
        };

        tracing::debug!("Took {:?} since last capture", last_capture.elapsed());

        let frame_data = {
            let instant = std::time::Instant::now();

            let framebuffer: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();
            let frame_metadata_status = framebuffer.metadata().unwrap().status();

            if frame_metadata_status != FrameMetadataStatus::Success {
                tracing::error!("Frame metadata status: {:?}", frame_metadata_status);
                req.reuse(ReuseFlag::REUSE_BUFFERS);

                // Re-queue only if there's still demand
                if coordinator.has_waiting_clients() {
                    cam.queue_request(req).map_err(|(_, e)| e)?;
                    requests_queued += 1;
                } else {
                    pending_requests.push(req);
                }
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
            let frame_data = planes.get(0).unwrap();
            tracing::debug!("Frame captured in {:?}", instant.elapsed());

            &frame_data[..bytes_used]
        };

        let convert_instant = std::time::Instant::now();
        camera_stream.convert_frame(&cfg_ref, frame_data, &mut buffer)?;
        tracing::debug!("YUYV->RGB conversion in {:?}", convert_instant.elapsed());

        let scale_instant = std::time::Instant::now();
        scale_rgb_bilinear(
            &buffer,
            &mut scaled,
            CAPTURE_WIDTH,
            CAPTURE_HEIGHT,
            OUTPUT_WIDTH,
            OUTPUT_HEIGHT,
        );
        tracing::debug!("Scaling in {:?}", scale_instant.elapsed());

        req.reuse(ReuseFlag::REUSE_BUFFERS);

        // Publish the frame to waiting clients
        coordinator.publish_frame(&scaled);
        tracing::debug!("Frame published to clients");

        last_capture = std::time::Instant::now();

        // Re-queue the request only if there's still demand
        if coordinator.has_waiting_clients() {
            cam.queue_request(req).map_err(|(_, e)| e)?;
            requests_queued += 1;
        } else {
            pending_requests.push(req);
            tracing::debug!("No more waiting clients, pausing capture");
        }
    }

    Ok(())
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
            // Use 4:2:0 subsampling for better compression
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
