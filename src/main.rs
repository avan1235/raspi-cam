use axum::response::Html;
use axum::{
    Router,
    extract::{
        State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    response::IntoResponse,
    routing::get,
};
use color_eyre::eyre::{Context, ContextCompat};
use futures_util::{SinkExt, StreamExt};
use libcamera::camera::CameraConfigurationStatus;
use libcamera::camera_manager::CameraManager;
use libcamera::framebuffer::AsFrameBuffer;
use libcamera::framebuffer::FrameMetadataStatus;
use libcamera::framebuffer_allocator::{FrameBuffer, FrameBufferAllocator};
use libcamera::framebuffer_map::MemoryMappedFrameBuffer;
use libcamera::geometry::Size;
use libcamera::pixel_format::PixelFormat;
use libcamera::request::ReuseFlag;
use libcamera::stream::StreamRole;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;
use tokio::sync::watch;
use turbojpeg::{Compressor, Image, PixelFormat as TJPixelFormat, Subsamp};

const OUTPUT_WIDTH: u32 = 640;
const OUTPUT_HEIGHT: u32 = 360;
const WEBSOCKET_ADDRESS: &str = "0.0.0.0:8080";
const JPEG_QUALITY: i32 = 75;

struct FrameState {
    jpg_data: Option<Vec<u8>>,
    frame_seq: u64,
}

#[derive(Clone)]
struct FrameCoordinator {
    state: Arc<Mutex<FrameState>>,
    frame_needed: Arc<Condvar>,
    waiting_clients: Arc<AtomicUsize>,
    frame_notify: watch::Sender<u64>,
    frame_receiver: watch::Receiver<u64>,
}

impl FrameCoordinator {
    fn new(_width: u32, _height: u32) -> Self {
        let (frame_notify, frame_receiver) = watch::channel(0u64);
        Self {
            state: Arc::new(Mutex::new(FrameState {
                jpg_data: None,
                frame_seq: 0,
            })),
            frame_needed: Arc::new(Condvar::new()),
            waiting_clients: Arc::new(AtomicUsize::new(0)),
            frame_notify,
            frame_receiver,
        }
    }

    fn has_waiting_clients(&self) -> bool {
        self.waiting_clients.load(Ordering::SeqCst) > 0
    }

    fn wait_for_demand(&self) {
        let state = self.state.lock().unwrap();

        let _guard = self
            .frame_needed
            .wait_while(state, |_| !self.has_waiting_clients())
            .unwrap();
    }

    fn publish_frame(&self, rgb_data: &[u8]) {
        if let Ok(jpeg_data) =
            encode_as_jpeg_turbo(&rgb_data, OUTPUT_WIDTH, OUTPUT_HEIGHT, JPEG_QUALITY)
        {
            let new_seq = {
                let mut state = self.state.lock().unwrap();
                state.frame_seq += 1;
                state.jpg_data = Some(jpeg_data.clone());
                state.frame_seq
            };
            let _ = self.frame_notify.send(new_seq);
        }
    }
}

#[derive(Clone)]
struct AppState {
    frame_coordinator: FrameCoordinator,
}

impl AppState {
    fn new(frame_coordinator: FrameCoordinator) -> Self {
        Self { frame_coordinator }
    }
}

fn yuyv_to_rgb(yuyv: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    let mut j = 0;

    for chunk in yuyv.chunks_exact(4) {
        let y0 = chunk[0] as i32 - 16;
        let u = chunk[1] as i32 - 128;
        let y1 = chunk[2] as i32 - 16;
        let v = chunk[3] as i32 - 128;

        let r_add = 409 * v + 128;
        let g_add = -100 * u - 208 * v + 128;
        let b_add = 516 * u + 128;

        let y0_scaled = 298 * y0;
        rgb[j] = ((y0_scaled + r_add) >> 8).clamp(0, 255) as u8;
        rgb[j + 1] = ((y0_scaled + g_add) >> 8).clamp(0, 255) as u8;
        rgb[j + 2] = ((y0_scaled + b_add) >> 8).clamp(0, 255) as u8;

        let y1_scaled = 298 * y1;
        rgb[j + 3] = ((y1_scaled + r_add) >> 8).clamp(0, 255) as u8;
        rgb[j + 4] = ((y1_scaled + g_add) >> 8).clamp(0, 255) as u8;
        rgb[j + 5] = ((y1_scaled + b_add) >> 8).clamp(0, 255) as u8;

        j += 6;
    }
    rgb
}

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let coordinator = FrameCoordinator::new(OUTPUT_WIDTH, OUTPUT_HEIGHT);
    let app_state = AppState::new(coordinator.clone());

    tokio::task::spawn_blocking(move || {
        let _ = run_camera_capture(coordinator);
    });

    let app = Router::new()
        .route("/video", get(ws_handler))
        .route("/", get(index_handler))
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind(WEBSOCKET_ADDRESS).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn index_handler() -> Html<&'static str> {
    Html(
        r#"<!DOCTYPE html>
<html lang="EN">
<head>
    <title>Camera</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
    <style>
        body {
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        img {
            max-width: 100%;
            max-height: 100%;
        }
    </style>
</head>
<body>
<img id="video-frame" alt="Waiting for stream..." src=""/>
<script>
    (function () {
        const camera = new WebSocket(`ws://${window.location.host}/video`);
        const img = document.getElementById("video-frame");
        let currentUrl = null;

        camera.binaryType = "blob";

        camera.onerror = () => { img.alt = "Camera WebSocket Error"; };

        camera.onmessage = (event) => {
            if (currentUrl) {
                URL.revokeObjectURL(currentUrl);
            }
            currentUrl = URL.createObjectURL(event.data);
            img.src = currentUrl;
        };
    })();
</script>
</body>
</html>"#,
    )
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_websocket(socket, state))
}

async fn handle_websocket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();

    state
        .frame_coordinator
        .waiting_clients
        .fetch_add(1, Ordering::SeqCst);

    state.frame_coordinator.frame_needed.notify_one();

    let mut rx = state.frame_coordinator.frame_receiver.clone();
    let mut last_seq = *rx.borrow_and_update();

    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Close(_) = msg {
                break;
            }
        }
    });

    loop {
        tokio::select! {
            _ = &mut recv_task => {
                break;
            }
            res = rx.changed() => {
                if res.is_err() {
                    break;
                }

                let new_seq = *rx.borrow();
                if new_seq > last_seq {
                    last_seq = new_seq;

                    let jpg_data = {
                        let lock = state.frame_coordinator.state.lock().unwrap();
                        lock.jpg_data.clone()
                    };

                    if let Some(jpg_data) = jpg_data {
                        if sender.send(Message::Binary(jpg_data.into())).await.is_err() {
                            break;
                        }
                    }
                }
            }
        }
    }

    state
        .frame_coordinator
        .waiting_clients
        .fetch_sub(1, Ordering::SeqCst);
}

fn run_camera_capture(coordinator: FrameCoordinator) -> color_eyre::Result<()> {
    let camera_manager = CameraManager::new()?;
    let cameras = camera_manager.cameras();
    let cam = cameras.get(0).expect("No cameras found");

    let mut cam = cam.acquire()?;

    let mut cfg = cam
        .generate_configuration(&[StreamRole::VideoRecording])
        .context("Failed to generate configuration")?;

    let mut stream_cfg = cfg.get_mut(0).unwrap();

    stream_cfg.set_size(Size::new(OUTPUT_WIDTH, OUTPUT_HEIGHT));
    stream_cfg.set_pixel_format(PixelFormat::new(u32::from_le_bytes(*b"RGB3"), 0));

    match cfg.validate() {
        CameraConfigurationStatus::Invalid => {
            color_eyre::eyre::bail!("Camera configuration invalid")
        }
        _ => {}
    }

    cam.configure(&mut cfg)
        .context("Unable to configure camera")?;

    let mut alloc = FrameBufferAllocator::new(&cam);
    let stream = cfg.get(0).unwrap().stream().unwrap();
    let buffers = alloc.alloc(&stream)?;

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

    let mut requests_queued = 0;
    let mut pending_requests: Vec<_> = reqs.into_iter().collect();

    loop {
        if requests_queued == 0 {
            coordinator.wait_for_demand();

            while let Some(req) = pending_requests.pop() {
                cam.queue_request(req).map_err(|(_, e)| e)?;
                requests_queued += 1;
            }
        }

        let mut req = match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(req) => {
                requests_queued -= 1;
                req
            }
            Err(_) => {
                continue;
            }
        };

        let framebuffer: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();

        if framebuffer.metadata().unwrap().status() == FrameMetadataStatus::Success {
            let bytes_used = framebuffer
                .metadata()
                .unwrap()
                .planes()
                .get(0)
                .unwrap()
                .bytes_used as usize;

            let planes = framebuffer.data();
            let frame_data = planes.get(0).unwrap();

            let rgb_data = if bytes_used == (OUTPUT_WIDTH * OUTPUT_HEIGHT * 2) as usize {
                yuyv_to_rgb(&frame_data[..bytes_used], OUTPUT_WIDTH, OUTPUT_HEIGHT)
            } else if bytes_used == (OUTPUT_WIDTH * OUTPUT_HEIGHT * 3) as usize {
                frame_data[..bytes_used].to_vec()
            } else {
                vec![]
            };

            if !rgb_data.is_empty() {
                coordinator.publish_frame(&rgb_data);
            }
        }

        req.reuse(ReuseFlag::REUSE_BUFFERS);

        if coordinator.has_waiting_clients() {
            cam.queue_request(req).map_err(|(_, e)| e)?;
            requests_queued += 1;
        } else {
            pending_requests.push(req);
        }
    }
}

thread_local! {
    static COMPRESSOR: std::cell::RefCell<Option<Compressor>> = const { std::cell::RefCell::new(None) };
}

fn encode_as_jpeg_turbo(
    rgb_data: &[u8],
    width: u32,
    height: u32,
    quality: i32,
) -> Result<Vec<u8>, String> {
    COMPRESSOR.with(|comp_cell| {
        let mut comp_opt = comp_cell.borrow_mut();

        if comp_opt.is_none() {
            let mut compressor = Compressor::new().map_err(|e| e.to_string())?;
            compressor.set_quality(quality).map_err(|e| e.to_string())?;
            compressor
                .set_subsamp(Subsamp::Sub2x2)
                .map_err(|e| e.to_string())?;
            *comp_opt = Some(compressor);
        }

        let compressor = comp_opt.as_mut().unwrap();

        let image = Image {
            pixels: rgb_data,
            width: width as usize,
            height: height as usize,
            pitch: (width * 3) as usize,
            format: TJPixelFormat::RGB,
        };

        compressor.compress_to_vec(image).map_err(|e| e.to_string())
    })
}