"""Live Desktop Camera Viewer for Campus Polyhouse CPPLUS NVR.

Connects to the RTSP stream and displays a native OpenCV desktop window.
Press 'q' or ESC on the video window to exit.
Press 's' to save a snapshot frame to docs/decisions/0008-campus-server/live_snapshot.jpg.
"""

from __future__ import annotations

import os
import sys
import urllib.parse
from pathlib import Path

import cv2
from dotenv import load_dotenv


def main() -> None:
    # Load environment variables
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    camera_url = os.environ.get("CAMPUS_CAMERA_URL", "")
    host = (
        camera_url.replace("http://", "").replace("https://", "").split(":")[0].strip()
    )
    user = os.environ.get("CAMPUS_SECONDARY_USER", "admin")
    password = os.environ.get("CAMPUS_SECONDARY_PASSWORD", "")

    if not host or not password:
        print("[ERROR] CAMPUS_CAMERA_URL or CAMPUS_SECONDARY_PASSWORD missing in .env")
        sys.exit(1)

    encoded_pwd = urllib.parse.quote(password)
    rtsp_url = (
        f"rtsp://{user}:{encoded_pwd}@{host}:554/cam/realmonitor?channel=1&subtype=1"
    )

    print("==================================================")
    print("  Connecting to Campus Polyhouse Camera Stream... ")
    print("  Subtype: 1 (Live Preview, 704x576)")
    print("  Controls on Video Window:")
    print("    [Q] or [ESC] -> Exit")
    print("    [S]          -> Save snapshot frame")
    print("==================================================")

    # Configure OpenCV capture with FFMPEG backend
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for real-time low latency

    if not cap.isOpened():
        print(
            "[FAIL] Could not connect to camera RTSP stream. Verify campus network connection."
        )
        sys.exit(1)

    window_name = "Polyhouse Camera - Live Stream (Press Q to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Stream interrupted or waiting for frames...")
                cv2.waitKey(100)
                continue

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                break
            elif key in (ord("s"), ord("S")):
                out_path = (
                    Path(__file__).resolve().parent.parent
                    / "docs"
                    / "decisions"
                    / "0008-campus-server"
                    / "live_snapshot.jpg"
                )
                cv2.imwrite(str(out_path), frame)
                print(f"[OK] Saved snapshot to {out_path.name}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Live viewer closed cleanly.")


if __name__ == "__main__":
    main()
