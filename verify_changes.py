import cv2
import numpy as np
from ui.hud_overlays import draw_hud, apply_lens
from core.vision_engine import apply_mode

def test_hud():
    # Test normal resolution
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = draw_hud(frame, "TARGET_HUD")
    assert out.shape == frame.shape
    print("Normal res HUD: OK")

    # Test low resolution (should hide some elements)
    low_res_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out_low = draw_hud(low_res_frame, "TARGET_HUD")
    assert out_low.shape == low_res_frame.shape
    print("Low res HUD: OK")

    # Test lens optimization
    lens_out = apply_lens(frame)
    assert lens_out.shape == frame.shape
    print("Lens optimization: OK")

    # Test mode without reticle
    thermal_out = draw_hud(frame, "THERMAL_VISION")
    # Visually inspecting would be better, but we can check if it runs without error
    print("Thermal HUD (no reticle): OK")

if __name__ == "__main__":
    test_hud()
