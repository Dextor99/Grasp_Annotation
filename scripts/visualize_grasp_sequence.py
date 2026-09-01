"""Interactive Open3D viewer for Top-1, Top-20 and high-quality grasps."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_visualization import (
    build_visualization_geometries,
    load_grasp_records,
    load_meta,
    select_grasps,
)


STATE_NAMES = ("top1", "top20", "high_quality")
DEFAULT_MODELS = (
    ("juxing", "model/juxing.ply", "results/ours-main/juxing"),
    ("yuanzhu", "model/yuanzhu.ply", "results/ours-main/yuanzhu"),
    ("sanjiao", "model/sanjiao.ply", "results/ours-main/sanjiao"),
    ("huixing", "model/huixing.ply", "results/ours-main/huixing"),
    ("shuilongtou", "model/shuilongtou.ply", "results/ours-main/shuilongtou"),
    ("cat", "model/colmap/cat.ply", "results/ours-main/cat"),
)


def parse_model_spec(spec):
    """Parse ``name=object_path=result_directory``."""
    text = str(spec)
    first_separator = text.find("=")
    last_separator = text.rfind("=")
    if first_separator <= 0 or last_separator <= first_separator + 1 or last_separator >= len(text) - 1:
        raise ValueError("--model must use name=object_path=result_directory")
    return {
        "name": text[:first_separator],
        "object": text[first_separator + 1:last_separator],
        "results": text[last_separator + 1:],
    }


def advance_sequence_index(model_index, state_index, model_count, state_count=3):
    """Advance state first, then model, wrapping after the final scene."""
    if model_count <= 0 or state_count <= 0:
        raise ValueError("model_count and state_count must be positive")
    next_state = state_index + 1
    if next_state < state_count:
        return model_index, next_state
    return (model_index + 1) % model_count, 0


def _model_records(model):
    records = load_grasp_records(model["results"])
    meta = load_meta(model["results"])
    return records, meta


def _selected_records(records, state_index, topk, score_threshold):
    ranked = select_grasps(records, topk=None, score_threshold=score_threshold)
    if state_index == 0:
        return ranked[:1]
    if state_index == 1:
        return ranked[:topk]
    return ranked


def show_interactive_sequence(models, topk=20, score_threshold=0.8, point_size=3.0, screenshot_dir=None):
    """Open a key-controlled Open3D viewer over all model/state scenes."""
    import open3d as o3d

    if not models:
        raise ValueError("at least one model is required")
    if topk <= 0 or point_size <= 0:
        raise ValueError("topk and point_size must be positive")
    if screenshot_dir is None:
        screenshot_dir = Path("results/interactive-screenshots")
    screenshot_dir = Path(screenshot_dir)

    prepared = []
    for model in models:
        records, meta = _model_records(model)
        prepared.append({"model": model, "records": records, "meta": meta})

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    if not visualizer.create_window(window_name="Grasp sequence viewer", width=1440, height=1000):
        raise RuntimeError("Open3D could not create the interactive window")
    current = {"model": 0, "state": 0, "geometries": []}

    def load_scene(model_index, state_index, reset_bounding_box=True):
        for geometry in current["geometries"]:
            visualizer.remove_geometry(geometry, reset_bounding_box=False)
        item = prepared[model_index]
        records = _selected_records(item["records"], state_index, topk, score_threshold)
        geometries, _ = build_visualization_geometries(
            item["model"]["object"],
            records,
            mode="overlay",
            style="line",
            color_by="score",
        )
        current["geometries"] = geometries
        for geometry in geometries:
            visualizer.add_geometry(geometry, reset_bounding_box=False)
        if reset_bounding_box:
            visualizer.reset_view_point(True)
        visualizer.poll_events()
        visualizer.update_renderer()
        current["model"] = model_index
        current["state"] = state_index
        state_name = STATE_NAMES[state_index]
        print(
            f"[{model_index + 1}/{len(prepared)}] {item['model']['name']} / "
            f"{state_name}: {len(records)} grasps"
        )

    def advance(_visualizer):
        model_index, state_index = advance_sequence_index(
            current["model"], current["state"], len(prepared), len(STATE_NAMES)
        )
        load_scene(model_index, state_index, reset_bounding_box=model_index != current["model"])
        return False

    def save_current(_visualizer):
        item = prepared[current["model"]]
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        path = screenshot_dir / f"{item['model']['name']}_{STATE_NAMES[current['state']]}.png"
        visualizer.poll_events()
        visualizer.update_renderer()
        if not visualizer.capture_screen_image(str(path), do_render=True):
            print(f"Failed to save screenshot: {path}")
        else:
            print(f"Saved screenshot: {path}")
        return False

    def close_viewer(_visualizer):
        close = getattr(visualizer, "close", None)
        if callable(close):
            close()
        return False

    visualizer.register_key_callback(ord(" "), advance)
    visualizer.register_key_callback(ord("S"), save_current)
    visualizer.register_key_callback(ord("Q"), close_viewer)
    visualizer.register_key_callback(256, close_viewer)  # Escape in GLFW/Open3D
    print("Controls: Space=next scene, S=save screenshot, Q/Esc=quit")
    try:
        load_scene(0, 0)
        visualizer.run()
    finally:
        visualizer.destroy_window()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        help="Repeatable name=object_path=result_directory; defaults to six main models",
    )
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.8)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--screenshot-dir", default="results/interactive-screenshots")
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    if args.topk <= 0 or args.point_size <= 0:
        raise ValueError("--topk and --point-size must be positive")
    if args.score_threshold != args.score_threshold:
        raise ValueError("--score-threshold must be finite")
    if args.model:
        models = [parse_model_spec(spec) for spec in args.model]
    else:
        models = [
            {"name": name, "object": object_path, "results": results}
            for name, object_path, results in DEFAULT_MODELS
        ]
    show_interactive_sequence(
        models,
        topk=args.topk,
        score_threshold=args.score_threshold,
        point_size=args.point_size,
        screenshot_dir=args.screenshot_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
