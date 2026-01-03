use std::str::FromStr;
use libcamera::camera::{Camera, CameraConfiguration};
use libcamera::pixel_format::PixelFormat;
use libcamera::stream::StreamConfigurationRef;
use ndi::{FourCCVideoType};
use super::{supports_configuration, CameraStream, FrameInfo};

pub struct MjpegStream;

impl CameraStream for MjpegStream {
    fn name(&self) -> &'static str {
        "mjpeg"
    }

    fn is_supported(&self, camera: &Camera) -> Option<CameraConfiguration> {
        supports_configuration(camera, PixelFormat::from_str("MJPEG").unwrap())
    }

    fn convert_frame(&self, cfg: &StreamConfigurationRef, frame: &[u8], target_buffer: &mut [u8]) -> color_eyre::Result<FrameInfo> {
        let rgba_stride = cfg.get_size().width * 4;

        let image = turbojpeg::decompress(frame, turbojpeg::PixelFormat::RGBX)?;

        target_buffer.copy_from_slice(&image.pixels);

        Ok(FrameInfo {
            video_type: FourCCVideoType::RGBX,
            stride: rgba_stride,
        })
    }
}