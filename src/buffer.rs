use std::ops::{Deref, DerefMut};
use libcamera::geometry::Size;

pub struct DoubleBuffer {
    buffer_a: Vec<u8>,
    buffer_b: Vec<u8>,
    index: usize,
}

impl DoubleBuffer {
    pub fn new(size: Size) -> Self {
        Self {
            buffer_a: vec![0; size.width as usize * size.height as usize * 4],
            buffer_b: vec![0; size.width as usize * size.height as usize * 4],
            index: 0,
        }
    }

    pub fn swap(&mut self) {
        (self.index, _) = self.index.overflowing_add(1);
    }
}

impl Deref for DoubleBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        if self.index % 2 == 0 {
            &self.buffer_a
        } else {
            &self.buffer_b
        }
    }
}

impl DerefMut for DoubleBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.index % 2 == 0 {
            &mut self.buffer_a
        } else {
            &mut self.buffer_b
        }
    }
}