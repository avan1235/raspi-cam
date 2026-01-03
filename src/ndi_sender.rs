use libcamera::geometry::Size;
use ndi::{FrameFormatType, VideoData};
use crate::FrameInfo;

pub struct NdiSender {
    sender: ndi::Send,
    width: i32,
    height: i32,
    fps: i32,
}

impl NdiSender {
    pub fn new(size: Size, fps: u32) -> color_eyre::Result<Self> {
        let sender = ndi::Send::new()?;

        Ok(Self {
            sender,
            width: size.width as i32,
            height: size.height as i32,
            fps: fps as i32,
        })
    }

    pub fn send(&self, buffer: &mut [u8], frame_info: &FrameInfo) -> color_eyre::Result<()> {
        let video_data = VideoData::from_buffer(
            self.width,
            self.height,
            frame_info.video_type,
            self.fps,
            1,
            FrameFormatType::Progressive,
            0,
            frame_info.stride as i32,
            None,
            buffer
        );
        self.sender.send_video_async(&video_data);

        Ok(())
    }
}