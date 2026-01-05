use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use clap::Parser;
use color_eyre::eyre::Context;
use image::{ImageBuffer, RgbaImage};
use libcamera::camera_manager::CameraManager;
use libcamera::*;
use libcamera::camera::{Camera, CameraConfiguration, CameraConfigurationStatus};
use libcamera::framebuffer::{AsFrameBuffer, FrameMetadataStatus};
use libcamera::framebuffer_allocator::{FrameBuffer, FrameBufferAllocator};
use libcamera::framebuffer_map::MemoryMappedFrameBuffer;
use libcamera::geometry::Size;
use libcamera::pixel_format::PixelFormat;
use libcamera::request::{ReuseFlag};
use libcamera::stream::{StreamConfigurationRef, StreamRole};
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use futures_util::{StreamExt, SinkExt};
use crate::buffer::DoubleBuffer;

mod yuyv;
mod buffer;

#[derive(Debug, Clone, Parser)]
#[command(version)]
pub struct Flags {
    #[arg(long, default_value_t = 1920)]
    pub width: u32,
    #[arg(long, default_value_t = 1080)]
    pub height: u32,
    #[arg(short, long, default_value_t = 60)]
    pub fps: u32,
    #[arg(short, long)]
    pub name: Option<String>,
    #[arg(long)]
    pub format: Option<String>,
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub websocket_address: String,
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

    fn get_png(&self) -> Option<Vec<u8>> {
        let data = self.data.lock().unwrap();
        data.as_ref().and_then(|rgba_data| {
            encode_as_png(rgba_data, self.width, self.height).ok()
        })
    }
}

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    let flags = Flags::parse();
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    let frame_buffer = SharedFrameBuffer::new(flags.width, flags.height);

    // Start WebSocket server
    let ws_addr = flags.websocket_address.clone();
    let frame_buffer_ws = frame_buffer.clone();
    tokio::spawn(async move {
        if let Err(e) = run_websocket_server(&ws_addr, frame_buffer_ws).await {
            tracing::error!("WebSocket server error: {}", e);
        }
    });

    // Run camera capture in blocking thread
    let flags_clone = flags.clone();
    tokio::task::spawn_blocking(move || {
        if let Err(e) = run_camera_capture(flags_clone, frame_buffer) {
            tracing::error!("Camera capture error: {}", e);
        }
    }).await?;

    Ok(())
}

async fn run_websocket_server(addr: &str, frame_buffer: SharedFrameBuffer) -> color_eyre::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("WebSocket server listening on: {}", addr);

    while let Ok((stream, addr)) = listener.accept().await {
        tracing::info!("New WebSocket connection from: {}", addr);
        let frame_buffer = frame_buffer.clone();
        tokio::spawn(handle_client(stream, frame_buffer));
    }

    Ok(())
}

async fn handle_client(stream: TcpStream, frame_buffer: SharedFrameBuffer) {
    let ws_stream = match accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            tracing::error!("Error during WebSocket handshake: {}", e);
            return;
        }
    };

    let (mut write, mut read) = ws_stream.split();

    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                if text.trim() == "s" {
                    tracing::debug!("Received 's' command, sending latest frame");

                    match frame_buffer.get_png() {
                        Some(png_data) => {
                            if let Err(e) = write.send(Message::Binary(png_data.into())).await {
                                tracing::error!("Error sending frame: {}", e);
                                break;
                            }
                        }
                        None => {
                            tracing::warn!("No frame available yet");
                            continue;
                        }
                    }
                }
            }
            Ok(Message::Close(_)) => {
                tracing::info!("Client disconnected");
                break;
            }
            Err(e) => {
                tracing::error!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }
}

fn run_camera_capture(
    flags: Flags,
    frame_buffer: SharedFrameBuffer,
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
    let camera_stream = if let Some(format_name) = flags.format {
        let format_name = format_name.to_lowercase();
        stream_formats.into_iter().find_map(|stream| {
            (stream.name() == format_name).then_some(()).and(stream.is_supported(&cam).map(|cfg| (stream, cfg)))
        })
    } else {
        stream_formats.into_iter().find_map(|stream| stream.is_supported(&cam).map(|cfg| (stream, cfg)))
    };

    let Some((camera_stream, mut cfg)) = camera_stream else {
        color_eyre::eyre::bail!("No supported stream format found");
    };

    cfg.get_mut(0).unwrap().set_size(Size::new(flags.width, flags.height));

    match cfg.validate() {
        CameraConfigurationStatus::Adjusted => tracing::warn!("Camera configuration was adjusted after changing frame size: {cfg:#?}"),
        CameraConfigurationStatus::Invalid => color_eyre::eyre::bail!("Error validating camera configuration after changing frame_size"),
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

    let mut buffer = DoubleBuffer::new(cfg_ref.get_size());
    let mut last_capture = std::time::Instant::now();

    loop {
        let mut req = rx.recv_timeout(Duration::from_secs(10))?;
        tracing::debug!("Took {:?} since last capture", last_capture.elapsed());

        let frame_data = {
            let instant = std::time::Instant::now();

            tracing::debug!("Camera request {req:?} completed!");
            tracing::trace!("Metadata: {:#?}", req.metadata());

            let framebuffer: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();
            tracing::trace!("FrameBuffer metadata: {:#?}", framebuffer.metadata());
            let frame_metadata_status = framebuffer.metadata().unwrap().status();
            if frame_metadata_status != FrameMetadataStatus::Success {
                tracing::error!("Frame metadata status: {:?}", frame_metadata_status);
                req.reuse(ReuseFlag::REUSE_BUFFERS);
                cam.queue_request(req).map_err(|(_, e)| e)?;
                continue;
            }
            let bytes_used = framebuffer.metadata().unwrap().planes().get(0).unwrap().bytes_used as usize;

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
}

fn encode_as_png(rgba_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, image::ImageError> {
    let img: RgbaImage = ImageBuffer::from_raw(width, height, rgba_data.to_vec())
        .expect("Invalid buffer size");

    let mut png_data = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut png_data);
    img.write_to(&mut cursor, image::ImageFormat::Png)?;

    Ok(png_data)
}

trait CameraStream {
    fn name(&self) -> &'static str;
    fn is_supported(&self, camera: &Camera) -> Option<CameraConfiguration>;
    fn convert_frame(&self, configuration: &StreamConfigurationRef, data: &[u8], target_buffer: &mut [u8]) -> color_eyre::Result<()>;
}

fn supports_configuration(cam: &Camera, format: PixelFormat) -> Option<CameraConfiguration> {
    let mut cfgs = cam.generate_configuration(&[StreamRole::VideoRecording])?;
    cfgs.get_mut(0)?.set_pixel_format(format);

    tracing::trace!("Generated config: {cfgs:#?}");

    match cfgs.validate() {
        CameraConfigurationStatus::Valid => tracing::debug!("Camera configuration {format} valid!"),
        CameraConfigurationStatus::Adjusted => tracing::trace!("Camera configuration was adjusted: {cfgs:#?}"),
        CameraConfigurationStatus::Invalid => tracing::trace!("Error validating camera configuration for {format}"),
    }

    if cfgs.get(0).unwrap().get_pixel_format() != format {
        return None;
    }

    Some(cfgs)
}