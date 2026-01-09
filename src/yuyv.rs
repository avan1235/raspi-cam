use super::{CameraStream, supports_configuration};
use color_eyre::eyre::ContextCompat;
use libcamera::camera::{Camera, CameraConfiguration};
use libcamera::color_space::{Range, YcbcrEncoding};
use libcamera::pixel_format::PixelFormat;
use libcamera::stream::StreamConfigurationRef;
use std::str::FromStr;
use yuvutils_rs::{YuvPackedImage, YuvRange, YuvStandardMatrix, yuyv422_to_rgb};

pub struct YuyvStream;

impl CameraStream for YuyvStream {
    fn is_supported(&self, camera: &Camera) -> Option<CameraConfiguration> {
        supports_configuration(camera, PixelFormat::from_str("YUYV").unwrap())
    }

    fn convert_frame(
        &self,
        cfg: &StreamConfigurationRef,
        frame: &[u8],
        target_buffer: &mut [u8],
    ) -> color_eyre::Result<()> {
        let rgb_stride = cfg.get_size().width * 3;

        let yuv_image = YuvPackedImage {
            height: cfg.get_size().height,
            width: cfg.get_size().width,
            yuy: frame,
            yuy_stride: cfg.get_stride(),
        };

        let color_space = cfg.get_color_space().context("No color space found")?;
        yuyv422_to_rgb(
            &yuv_image,
            target_buffer,
            rgb_stride,
            match color_space.range {
                Range::Full => YuvRange::Full,
                Range::Limited => YuvRange::Limited,
            },
            match color_space.ycbcr_encoding {
                YcbcrEncoding::Rec601 => YuvStandardMatrix::Bt601,
                YcbcrEncoding::Rec709 => YuvStandardMatrix::Bt709,
                YcbcrEncoding::Rec2020 => YuvStandardMatrix::Bt2020,
                YcbcrEncoding::None => {
                    return Err("Unknown ycbcr_encoding, choose your default and put here".into());
                }
            },
        )?;

        Ok(())
    }
}
