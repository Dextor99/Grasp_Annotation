# Independent evaluation asset derived from `model/2.stl`

The original STL is preserved and is never overwritten.  Its dimensions are
interpreted as millimetres and converted once to metres in `2_scaled.obj`.
`2_repaired.obj` is a separately written watertight copy produced by the
non-destructive preparation script; `2_repaired.sdf` is a 64³ SDF generated
from that copy.  `original_metadata.json` and `repair_report.json` record the
source state and repair method for auditability.

This asset is suitable for pipeline smoke tests.  It must not be described as
an exact repair of the source geometry without a visual/manual inspection.
