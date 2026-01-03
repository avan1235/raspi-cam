use std::ops::{Deref, DerefMut};
use std::time::Duration;
use clap::Parser;
use color_eyre::eyre::Context;
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
use ndi::{FourCCVideoType, FrameFormatType, VideoData};
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;
use crate::buffer::DoubleBuffer;

mod yuyv;
mod buffer;
mod ndi_sender;

#[derive(Debug, Clone, Parser)]
#[command(version)]
pub struct Flags {
    #[arg(long, default_value_t = 1920)]
    pub width: u32,
    #[arg(long, default_value_t = 1080)]
    pub height: u32,
    #[arg(short, long, default_value_t = 60)]
    pub fps: u32, // TODO: check for fps in stream configuration
    #[arg(short, long)]
    pub name: Option<String>,
    #[arg(long)]
    pub format: Option<String>,
}

fn main() -> color_eyre::Result<()> {
    let stream_formats: Vec<Box<dyn CameraStream>> = vec![
        Box::new(yuyv::YuyvStream),
    ];

    ndi::initialize()?;
    let flags = Flags::parse();
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    let camera_manager = CameraManager::new()?;
    let cameras = camera_manager.cameras();

    let cam = cameras.get(0).expect("No cameras found");

    tracing::info!(
        "Using camera: {}",
        *cam.properties().get::<properties::Model>()?
    );

    let mut cam = cam.acquire()?;

    let camera_stream = if let Some(format_name) = flags.format {
        let format_name = format_name.to_lowercase();
        stream_formats.into_iter().find_map(|stream| (stream.name() == format_name).then_some(()).and(stream.is_supported(&cam).map(|cfg| (stream, cfg))))
    }else {
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

    // Allocate frame buffers for the stream
    let cfg = cfg.get(0).unwrap();
    let stream = cfg.stream().unwrap();
    let buffers = alloc.alloc(&stream)?;
    tracing::debug!("Allocated {} buffers", buffers.len());

    // Convert FrameBuffer to MemoryMappedFrameBuffer, which allows reading &[u8]
    let buffers = buffers
        .into_iter()
        .map(|buf| MemoryMappedFrameBuffer::new(buf).unwrap())
        .collect::<Vec<_>>();

    // Create capture requests and attach buffers
    let reqs = buffers
        .into_iter()
        .enumerate()
        .map(|(i, buf)| {
            let mut req = cam.create_request(Some(i as u64)).unwrap();
            req.add_buffer(&stream, buf).unwrap();
            req
        })
        .collect::<Vec<_>>();

    // Completed capture requests are returned as a callback
    let (tx, rx) = std::sync::mpsc::channel();
    cam.on_request_completed(move |req| {
        tx.send(req).unwrap();
    });

    // TODO: Set `Control::FrameDuration()` here. Blocked on https://github.com/lit-robotics/libcamera-rs/issues/2
    cam.start(None)?;

    // Enqueue all requests to the camera
    for req in reqs {
        tracing::debug!("Request queued for execution: {req:#?}");
        cam.queue_request(req).map_err(|(_, e)| e)?;
    }

    let ndi_sender = ndi_sender::NdiSender::new(cfg.get_size(), flags.fps)?;

    let mut buffer = DoubleBuffer::new(cfg.get_size());

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

        let frame_info = {
            let instant = std::time::Instant::now();
            // TODO: Can we run this conversion and sending to ndi in the background as well to reduce latency?
            let frame_info = camera_stream.convert_frame(&cfg, frame_data, &mut buffer)?;
            tracing::debug!("Converted to {:?} in {:?}", frame_info.video_type, instant.elapsed());

            frame_info
        };

        req.reuse(ReuseFlag::REUSE_BUFFERS);
        cam.queue_request(req).map_err(|(_, e)| e)?;

        {
            let instant = std::time::Instant::now();
            ndi_sender.send(&mut buffer, &frame_info)?;
            tracing::debug!("Sent to NDI in {:?}", instant.elapsed());
        }

        last_capture = std::time::Instant::now();
        buffer.swap();
    }
}

trait CameraStream {
    fn name(&self) -> &'static str;

    fn is_supported(&self, camera: &Camera) -> Option<CameraConfiguration>;

    fn convert_frame(&self, configuration: &StreamConfigurationRef, data: &[u8], target_buffer: &mut [u8]) -> color_eyre::Result<FrameInfo>;
}

pub struct FrameInfo {
    pub video_type: FourCCVideoType,
    pub stride: u32,
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