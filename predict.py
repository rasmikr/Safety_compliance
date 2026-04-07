"""
predict.py — Run PPE detection inference on images, videos, or webcam.

Usage:
    python predict.py --source image.jpg                         # single image
    python predict.py --source video.mp4 --save                  # save annotated video
    python predict.py --source 0                                 # webcam
    python predict.py --source path/to/folder --conf 0.4 --save  # folder of images
    python predict.py --source video.mp4 --frame-skip 5 --save   # process every 5th frame
"""

import argparse
import sys
from pathlib import Path

import cv2

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from config.settings import (
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_DEVICE,
    DEFAULT_PROJECT,
    DEFAULT_FRAME_SKIP,
    CLASS_NAMES,
)
from utils.visualization import annotate_frame, generate_compliance_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PPE detection inference",
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Image, video, directory, or webcam index (0)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model weights (.pt). "
                             "If not specified, looks for best.pt in runs/detect/ppe_detection/weights/")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"Confidence threshold (default: {DEFAULT_CONF})")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help=f"Inference image size (default: {DEFAULT_IMGSZ})")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                        help="Device for inference")
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP,
                        help=f"Process every Nth frame for video (default: {DEFAULT_FRAME_SKIP})")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated results to disk")
    parser.add_argument("--show", action="store_true",
                        help="Display results in a window (requires GUI)")
    parser.add_argument("--save-txt", action="store_true",
                        help="Save detection results as .txt files")
    return parser.parse_args()


def _resolve_model_path(model_arg: str | None) -> str:
    """Resolve the model weights path."""
    if model_arg:
        return model_arg

    # Default: look for trained weights
    default_weights = Path(DEFAULT_PROJECT) / "ppe_detection" / "weights" / "best.pt"
    if default_weights.exists():
        return str(default_weights)

    # Fallback: use pretrained yolo26n
    print("⚠️  No trained weights found. Using pretrained yolo26n.pt (not fine-tuned for PPE).")
    return "yolo26n.pt"


def _is_video_source(source: str) -> bool:
    """Check if the source is a video file or webcam."""
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    if source.isdigit():
        return True  # webcam
    return Path(source).suffix.lower() in video_exts


def _run_video_inference(model, args):
    """Run inference on video with frame skipping."""
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Cannot open video source: {args.source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Video FPS   : {fps:.1f}")
    print(f"  Resolution  : {width}x{height}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Frame Skip  : every {args.frame_skip} frame(s)")
    print(f"  Frames to process: ~{total_frames // args.frame_skip}")
    print("=" * 60)

    # Set up video writer if saving
    writer = None
    if args.save:
        out_dir = Path(DEFAULT_PROJECT) / "ppe_predict"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"annotated_{Path(str(args.source)).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps / args.frame_skip, (width, height))

    frame_idx = 0
    processed = 0
    violations = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.frame_skip == 0:
            # Run detection on this frame
            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device if args.device else None,
                verbose=False,
            )

            result = results[0]
            summary = generate_compliance_summary(result)
            status = "✅ COMPLIANT" if summary["is_compliant"] else "❌ VIOLATION"
            processed += 1

            if not summary["is_compliant"]:
                violations += 1

            print(f"  Frame {frame_idx:>6d} | {status} | Persons: {summary['total_persons']}", end="")
            if summary["worn_ppe"]:
                print(f" | Worn: {summary['worn_ppe']}", end="")
            if summary["missing_ppe"]:
                print(f" | Missing: {summary['missing_ppe']}", end="")
            print()

            # Annotate and optionally save/show
            annotated = annotate_frame(frame, result, conf_threshold=args.conf)

            if writer:
                writer.write(annotated)
            if args.show:
                cv2.imshow("PPE Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n⏹ Stopped by user.")
                    break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"\n   Annotated video saved to: {out_path}")
    if args.show:
        cv2.destroyAllWindows()

    print(f"\n{'=' * 60}")
    print(f"  Video Summary")
    print(f"{'=' * 60}")
    print(f"  Total frames   : {frame_idx}")
    print(f"  Frames processed: {processed}")
    print(f"  Violations     : {violations}")
    print(f"  Compliance rate: {((processed - violations) / max(processed, 1)) * 100:.1f}%")
    print(f"{'=' * 60}")


def _run_image_inference(model, args):
    """Run inference on images (single file or directory)."""
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        save=args.save,
        show=args.show,
        save_txt=args.save_txt,
        project=DEFAULT_PROJECT,
        name="ppe_predict",
        exist_ok=True,
    )

    # Print compliance summary for each result
    for i, result in enumerate(results):
        summary = generate_compliance_summary(result)
        status = "✅ COMPLIANT" if summary["is_compliant"] else "❌ VIOLATION DETECTED"

        print(f"\n--- Image {i + 1} ---")
        print(f"  Status     : {status}")
        print(f"  Persons    : {summary['total_persons']}")

        if summary["worn_ppe"]:
            print(f"  Worn PPE   : {summary['worn_ppe']}")
        if summary["missing_ppe"]:
            print(f"  Missing PPE: {summary['missing_ppe']}")

    print(f"\n✅ Inference complete! Processed {len(results)} image(s).")

    if args.save:
        print(f"   Results saved to: {DEFAULT_PROJECT}/ppe_predict/")


def main():
    args = parse_args()
    model_path = _resolve_model_path(args.model)

    print("=" * 60)
    print("  PPE Detection — Inference")
    print("=" * 60)
    print(f"  Model      : {model_path}")
    print(f"  Source      : {args.source}")
    print(f"  Confidence  : {args.conf}")
    print(f"  Image Size  : {args.imgsz}")

    # Load model
    model = YOLO(model_path)

    if _is_video_source(args.source):
        print(f"  Mode        : Video (frame skip = {args.frame_skip})")
        print("=" * 60)
        _run_video_inference(model, args)
    else:
        print("  Mode        : Image")
        print("=" * 60)
        _run_image_inference(model, args)

# from ultralytics import YOLO
# m = YOLO("runs/detect/ppe_detection/weights/best.pt")
# print(m.names)

if __name__ == "__main__":
    main()
