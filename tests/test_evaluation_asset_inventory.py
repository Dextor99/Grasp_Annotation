from scripts.common_eval.prepare_evaluation_assets import classify_asset


def test_asset_classification_prioritizes_watertight_mesh_then_repair_then_cloud():
    assert classify_asset(source_exists=True, watertight=True, sdf_exists=True) == "A_ready"
    assert classify_asset(source_exists=True, watertight=False, sdf_exists=False) == "B_repair_and_sdf"
    assert classify_asset(source_exists=False, watertight=False, sdf_exists=False) == "C_reference_required"

