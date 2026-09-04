# CI sample clip provenance (§12.5)

File: `data/ci_sample_clip.mp4`

This clip is **synthetic**. It was generated locally with ffmpeg from the
`lavfi` `color` source (solid black 1280×720, 2 seconds at 30 fps, plus a
drawn box and the text `CI Test`). It contains **no camera footage and no
identifiable people**.

It is used only for GitHub Actions and local integration tests
(`configs/ci.yaml`) to prove the pipeline wires together, not for published
evaluation metrics (those use `configs/default.yaml`).

Because it is computer-generated and not derived from a copyrighted film or
photograph, it is suitable for this public academic repository (treated as
public-domain / CC0-equivalent generated media).
