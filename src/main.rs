use crate::buffer::DoubleBuffer;
use color_eyre::eyre::Context;
use futures_util::{SinkExt, StreamExt};
use libcamera::camera::{Camera, CameraConfiguration, CameraConfigurationStatus};
use libcamera::camera_manager::CameraManager;
use libcamera::framebuffer::AsFrameBuffer;
use libcamera::framebuffer::{FrameMetadataStatus};
use libcamera::framebuffer_allocator::{FrameBuffer, FrameBufferAllocator};
use libcamera::framebuffer_map::MemoryMappedFrameBuffer;
use libcamera::geometry::Size;
use libcamera::pixel_format::PixelFormat;
use libcamera::request::ReuseFlag;
use libcamera::stream::{StreamConfigurationRef, StreamRole};
use libcamera::*;
use std::ops::Deref;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use turbojpeg::{Compressor, Image, PixelFormat as TJPixelFormat, Subsamp};
use crate::yuyv::YuyvStream;

mod buffer;
mod yuyv;

// Capture at native resolution for better quality source
const CAPTURE_WIDTH: u32 = 1920;
const CAPTURE_HEIGHT: u32 = 1080;

// Output resolution (720p) - optimized for Pi 3B network bandwidth
const OUTPUT_WIDTH: u32 = 640;
const OUTPUT_HEIGHT: u32 = 360;

const WEBSOCKET_ADDRESS: &str = "0.0.0.0:8080";

// Reduced quality for Pi 3B - balances quality vs encoding speed
const JPEG_QUALITY: i32 = 75;

// Shared state for the latest frame
#[derive(Clone)]
struct SharedFrameBuffer {
    // Stores the scaled RGB data ready for JPEG encoding
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

    fn update(&self, frame_data: Vec<u8>) {
        let mut data = self.data.lock().unwrap();
        *data = Some(frame_data);
    }

    fn get_rgb(&self) -> Option<Vec<u8>> {
        let data = self.data.lock().unwrap();
        data.clone()
    }
}

/// Scales RGB image using bilinear interpolation
/// Optimized for Raspberry Pi 3B with integer arithmetic where possible
fn scale_rgb_bilinear(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Vec<u8> {
    let mut dst = vec![0u8; (dst_width * dst_height * 3) as usize];

    // Pre-calculate scaling factors using fixed-point arithmetic (16.16)
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

    dst
}

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    // Output buffer uses scaled dimensions
    let frame_buffer = SharedFrameBuffer::new(OUTPUT_WIDTH, OUTPUT_HEIGHT);

    // Start WebSocket server
    let frame_buffer_ws = frame_buffer.clone();
    tokio::spawn(async move {
        if let Err(e) = run_websocket_server(WEBSOCKET_ADDRESS, frame_buffer_ws).await {
            tracing::error!("WebSocket server error: {}", e);
        }
    });

    tokio::task::spawn_blocking(move || {
        if let Err(e) = run_camera_capture(frame_buffer) {
            tracing::error!("Camera capture error: {}", e);
        }
    })
        .await?;

    Ok(())
}

async fn run_websocket_server(
    addr: &str,
    frame_buffer: SharedFrameBuffer,
) -> color_eyre::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("WebSocket server listening on: {}", addr);

    while let Ok((stream, addr)) = listener.accept().await {
        tracing::info!("New WebSocket connection from: {}", addr);
        let frame_buffer = frame_buffer.clone();
        tokio::spawn(handle_client(stream, frame_buffer, addr));
    }

    Ok(())
}

async fn handle_client(
    stream: TcpStream,
    frame_buffer: SharedFrameBuffer,
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

                    let rgb_data = match frame_buffer.get_rgb() {
                        Some(data) => data,
                        None => {
                            tracing::warn!("No frame available yet for client {}", client_addr);
                            continue;
                        }
                    };

                    let width = frame_buffer.width;
                    let height = frame_buffer.height;

                    let encode_start = std::time::Instant::now();
                    let jpeg_data = match tokio::task::spawn_blocking(move || {
                        encode_as_jpeg_turbo(&rgb_data, width, height, JPEG_QUALITY)
                    })
                        .await
                    {
                        Ok(Ok(data)) => data,
                        Ok(Err(e)) => {
                            tracing::error!("JPEG encoding error for client {}: {}", client_addr, e);
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
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) | Ok(Message::Binary(_)) | Ok(Message::Frame(_)) => {}
            Err(e) => {
                tracing::error!("WebSocket error with {}: {}", client_addr, e);
                break;
            }
        }
    }

    tracing::debug!("Connection handler for {} terminated", client_addr);
}

fn run_camera_capture(frame_buffer: SharedFrameBuffer) -> color_eyre::Result<()> {
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

    // Capture at full resolution for better quality source
    cfg.get_mut(0).unwrap().set_size(Size::new(CAPTURE_WIDTH, CAPTURE_HEIGHT));

    match cfg.validate() {
        CameraConfigurationStatus::Adjusted => {
            tracing::warn!("Camera configuration was adjusted: {cfg:#?}")
        }
        CameraConfigurationStatus::Invalid => {
            color_eyre::eyre::bail!("Error validating camera configuration")
        }
        _ => {}
    }

    cam.configure(&mut cfg).context("Unable to configure camera")?;

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

    for req in reqs {
        tracing::debug!("Request queued for execution: {req:#?}");
        cam.queue_request(req).map_err(|(_, e)| e)?;
    }

    // Buffer for full-resolution RGB conversion
    let mut buffer = DoubleBuffer::new(cfg_ref.get_size());
    let mut last_capture = std::time::Instant::now();

    // Pre-allocate scaling buffer to avoid repeated allocations
    let scaled_buffer_size = (OUTPUT_WIDTH * OUTPUT_HEIGHT * 3) as usize;

    loop {
        let mut req = rx.recv_timeout(Duration::from_secs(10))?;
        tracing::debug!("Took {:?} since last capture", last_capture.elapsed());

        let frame_data = {
            let instant = std::time::Instant::now();

            let framebuffer: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();
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
            let frame_data = planes.get(0).unwrap();
            tracing::debug!("Frame captured in {:?}", instant.elapsed());

            &frame_data[..bytes_used]
        };

        // Convert YUYV to RGB at full resolution
        let convert_instant = std::time::Instant::now();
        camera_stream.convert_frame(&cfg_ref, frame_data, &mut buffer)?;
        tracing::debug!("YUYV->RGB conversion in {:?}", convert_instant.elapsed());

        // Scale down to 720p
        let scale_instant = std::time::Instant::now();
        let scaled_rgb = scale_rgb_bilinear(
            buffer.deref(),
            CAPTURE_WIDTH,
            CAPTURE_HEIGHT,
            OUTPUT_WIDTH,
            OUTPUT_HEIGHT,
        );
        tracing::debug!("Scaling to 720p in {:?}", scale_instant.elapsed());

        req.reuse(ReuseFlag::REUSE_BUFFERS);
        cam.queue_request(req).map_err(|(_, e)| e)?;

        // Update shared frame buffer with scaled data
        frame_buffer.update(scaled_rgb);

        last_capture = std::time::Instant::now();
        buffer.swap();
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