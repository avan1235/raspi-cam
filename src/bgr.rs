use std::str::FromStr;
use libcamera::camera::{Camera, CameraConfiguration};
use libcamera::pixel_format::PixelFormat;
use libcamera::stream::StreamConfigurationRef;
use ndi::{FourCCVideoType};
use yuvutils_rs::bgr_to_bgra;
use super::{supports_configuration, CameraStream, FrameInfo};

pub struct BgrStream;

impl CameraStream for BgrStream {
    fn name(&self) -> &'static str {
        "bgr"
    }

    fn is_supported(&self, camera: &Camera) -> Option<CameraConfiguration> {
        supports_configuration(camera, PixelFormat::from_str("BGR888").unwrap())
    }

    fn convert_frame(&self, cfg: &StreamConfigurationRef, frame: &[u8], target_buffer: &mut [u8]) -> color_eyre::Result<FrameInfo> {
        let bgr_stride = cfg.get_size().width * 3;
        let bgra_stride = cfg.get_size().width * 4;

        bgr_to_bgra(frame, bgr_stride, target_buffer, bgra_stride, cfg.get_size().width, cfg.get_size().height)?;

        Ok(FrameInfo {
            video_type: FourCCVideoType::BGRX,
            stride: bgra_stride,
        })
    }
}