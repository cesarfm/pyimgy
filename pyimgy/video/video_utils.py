from contextlib import contextmanager
from pathlib import Path
from typing import Any, Tuple

import cv2


@contextmanager
def cv_reader(path: Path):
    reader = cv2.VideoCapture(str(path))
    try:
        yield reader
    finally:
        reader.release()


@contextmanager
def cv_writer(path: Path, fourcc: int, fps: float, frame_size):
    writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    try:
        yield writer
    finally:
        writer.release()


def get_frame_step(fps: float, extract_fps: float) -> int:
    return max(int(fps / extract_fps), 1)


def extract_next_frame(path: Path, extract_fps: float) -> Tuple[int, Any]:
    with cv_reader(path) as r:
        # we want to get the frame rate of the video
        fps = r.get(cv2.CAP_PROP_FPS)
        frame_step = get_frame_step(fps, extract_fps) if extract_fps is not None else 1

        print(f'Extracting frames from {path.name} with FPS={fps}, extraction fps={extract_fps}, frame step={frame_step}')

        while r.isOpened():
            frame_nr = int(r.get(cv2.CAP_PROP_POS_FRAMES))
            success, image = r.read()
            if not success:
                break
            if frame_nr % frame_step == 0:
                yield frame_nr, image


def extract_frames_from_video(video_fn: Path, save_path: Path, extract_fps: float = None, reset_indexes: bool = False) -> list:
    save_path.mkdir(parents=True, exist_ok=True)

    def _processor(idx, frame_nr, frame):
        fn = save_path / f'{(idx if reset_indexes else frame_nr):08}.png'
        cv2.imwrite(str(fn), frame)
        return frame_nr, fn

    return [_processor(idx, frame_nr, frame) for idx, (frame_nr, frame) in enumerate(extract_next_frame(video_fn, extract_fps))]
