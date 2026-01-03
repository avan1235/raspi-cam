use std::str::FromStr;
use libcamera::camera::{Camera, CameraConfiguration};
use libcamera::color_space::Range;
use libcamera::pixel_format::PixelFormat;
use libcamera::stream::StreamConfigurationRef;
use ndi::{FourCCVideoType};
use yuvutils_rs::{yuyv422_to_rgba, YuvPackedImage, YuvRange, YuvStandardMatrix};
use super::{supports_configuration, CameraStream, FrameInfo};

pub struct YuyvStream;

impl CameraStream for YuyvStream {
    fn name(&self) -> &'static str {
        "yuyv"
    }

    fn is_supported(&self, camera: &Camera) -> Option<CameraConfiguration> {
        supports_configuration(camera, PixelFormat::from_str("YUYV").unwrap())
    }

    fn convert_frame(&self, cfg: &StreamConfigurationRef, frame: &[u8], target_buffer: &mut [u8]) -> color_eyre::Result<FrameInfo> {
        let rgb_stride = cfg.get_size().width * 4;

        let yuv_image = YuvPackedImage {
            height: cfg.get_size().height,
            width: cfg.get_size().width,
            yuy: frame,
            yuy_stride: cfg.get_stride(),
        };

        yuyv422_to_rgba(
            &yuv_image,
            target_buffer,
            rgb_stride,
            match cfg.get_color_space().map(|cs| cs.range).unwrap_or(Range::Limited) {
                Range::Full => YuvRange::Full,
                Range::Limited => YuvRange::Limited,
            },
            YuvStandardMatrix::Bt709, // TODO: read from cfg
        )?;

        Ok(FrameInfo {
            video_type: FourCCVideoType::RGBX,
            stride: rgb_stride,
        })
    }
}