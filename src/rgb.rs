use std::str::FromStr;
use libcamera::camera::{Camera, CameraConfiguration};
use libcamera::pixel_format::PixelFormat;
use libcamera::stream::StreamConfigurationRef;
use ndi::{FourCCVideoType};
use yuvutils_rs::rgb_to_rgba;
use super::{supports_configuration, CameraStream, FrameInfo};

pub struct RgbStream;

impl CameraStream for RgbStream {
    fn name(&self) -> &'static str {
        "rgb"
    }

    fn is_supported(&self, camera: &Camera) -> Option<CameraConfiguration> {
        supports_configuration(camera, PixelFormat::from_str("RGB888").unwrap())
    }

    fn convert_frame(&self, cfg: &StreamConfigurationRef, frame: &[u8], target_buffer: &mut [u8]) -> color_eyre::Result<FrameInfo> {
        let rgb_stride = cfg.get_size().width * 3;
        let rgba_stride = cfg.get_size().width * 4;

        rgb_to_rgba(frame, rgb_stride, target_buffer, rgba_stride, cfg.get_size().width, cfg.get_size().height)?;

        Ok(FrameInfo {
            video_type: FourCCVideoType::RGBX,
            stride: rgba_stride,
        })
    }
}