use std::time::Duration;

use libcamera::{
    camera::CameraConfigurationStatus,
    camera_manager::CameraManager,
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    pixel_format::PixelFormat,
    properties,
    stream::StreamRole,
};
use yuv::{yuv420_to_rgb, YuvChromaSubsampling, YuvPlanarImageMut, YuvRange, YuvStandardMatrix};

// YUYV pixel format (YUV 4:2:2)
const PIXEL_FORMAT_YUYV: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'Y', b'U', b'Y', b'V']), 0);

fn main() {
    let filename = std::env::args().nth(1).expect("Usage ./raspi-cam <filename.jpg>");

    let mgr = CameraManager::new().unwrap();

    let cameras = mgr.cameras();

    println!("Found {} cameras, using first one", cameras.len());

    let cam = cameras.get(0).expect("No cameras found");

    println!(
        "Using camera: {}",
        *cam.properties().get::<properties::Model>().unwrap()
    );

    let mut cam = cam.acquire().expect("Unable to acquire camera");

    // This will generate default configuration for each specified role
    let mut cfgs = cam.generate_configuration(&[StreamRole::ViewFinder]).unwrap();

    // Use YUYV format since MJPEG is not supported
    cfgs.get_mut(0).unwrap().set_pixel_format(PIXEL_FORMAT_YUYV);

    println!("Generated config: {cfgs:#?}");

    match cfgs.validate() {
        CameraConfigurationStatus::Valid => println!("Camera configuration valid!"),
        CameraConfigurationStatus::Adjusted => println!("Camera configuration was adjusted: {cfgs:#?}"),
        CameraConfigurationStatus::Invalid => panic!("Error validating camera configuration"),
    }

    cam.configure(&mut cfgs).expect("Unable to configure camera");

    let mut alloc = FrameBufferAllocator::new(&cam);

    // Allocate frame buffers for the stream
    let cfg = cfgs.get(0).unwrap();
    let stream = cfg.stream().unwrap();
    let buffers = alloc.alloc(&stream).unwrap();
    println!("Allocated {} buffers", buffers.len());

    // Get the actual width and height from the configuration
    let width = cfg.get_size().width as usize;
    let height = cfg.get_size().height as usize;
    println!("Capture size: {}x{}", width, height);

    // Convert FrameBuffer to MemoryMappedFrameBuffer, which allows reading &[u8]
    let buffers = buffers
        .into_iter()
        .map(|buf| MemoryMappedFrameBuffer::new(buf).unwrap())
        .collect::<Vec<_>>();

    // Create capture requests and attach buffers
    let mut reqs = buffers
        .into_iter()
        .map(|buf| {
            let mut req = cam.create_request(None).unwrap();
            req.add_buffer(&stream, buf).unwrap();
            req
        })
        .collect::<Vec<_>>();

    // Completed capture requests are returned as a callback
    let (tx, rx) = std::sync::mpsc::channel();
    cam.on_request_completed(move |req| {
        tx.send(req).unwrap();
    });

    cam.start(None).unwrap();

    // Multiple requests can be queued at a time, but for this example we just want a single frame.
    cam.queue_request(reqs.pop().unwrap()).map_err(|(_, e)| e).unwrap();

    println!("Waiting for camera request execution");
    // Allow a bit more time for first exposure/conversion to complete on slower cameras.
    let req = rx.recv_timeout(Duration::from_secs(5)).expect("Camera request failed");

    println!("Camera request {req:?} completed!");
    println!("Metadata: {:#?}", req.metadata());

    // Get framebuffer for our stream
    let framebuffer: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();
    println!("FrameBuffer metadata: {:#?}", framebuffer.metadata());

    // YUYV format has one data plane
    let planes = framebuffer.data();
    let yuyv_data = planes.first().unwrap();
    let bytes_used = framebuffer.metadata().unwrap().planes().get(0).unwrap().bytes_used as usize;

    println!("Converting YUYV to RGB...");

    let mut yuv_planar_image_mut =
        YuvPlanarImageMut::<u8>::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv420);
    let yuv_planar_image = yuv_planar_image_mut.to_fixed();

    let mut rgba = vec![0u8; width * height * 4];
    let rgba_stride = width * 4;

    yuv420_to_rgb(
        &yuv_planar_image,
        &mut rgba,
        rgba_stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
        .unwrap();

    println!("Encoding JPEG...");
    // Encode RGB data to JPEG using the image crate
    let img = image::RgbaImage::from_raw(width as u32, height as u32, rgba)
        .expect("Failed to create image from RGB data");

    img.save(&filename).expect("Failed to save JPEG");

    println!("Written image to {}", &filename);
}