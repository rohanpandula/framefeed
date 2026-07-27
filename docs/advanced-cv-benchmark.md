# Advanced CV backend benchmark

This benchmark compares FrameFeed's YOLO26s foreground-person detector on an
NVIDIA RTX A4000 and an Intel UHD Graphics 770 integrated GPU. It is intended
to answer a practical deployment question: can Advanced CV move to the Intel
iGPU without materially changing photo crops?

## Test system

- CPU: Intel Core i9-12900K
- Integrated GPU: Intel UHD Graphics 770
- Discrete GPU: NVIDIA RTX A4000 16 GB
- Corpus: 340 family photos
- Model: YOLO26s person detection
- Input size: 1280 pixels
- Confidence threshold: 0.18
- Workload: sequential, single-photo inference
- CUDA runtime: Ultralytics 8.4.107
- Intel runtime: OpenVINO 2026.2.1, FP16 export, `GPU.0`
- Passes: two per primary backend after one warm-up image

The benchmark used [`scripts/benchmark_people.py`](../scripts/benchmark_people.py).
It includes image decoding, EXIF orientation, preprocessing, inference, and
postprocessing in total latency. Model loading is measured separately. No source
photos or detection metadata are stored in the repository.

## Performance

| Backend | Input geometry | Corpus pass | Throughput | Median inference | Median total |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX A4000 CUDA | Rectangular, current service behavior | 18.05–18.36 s | 18.5–18.8 photos/s | 11.23 ms | 51.97 ms |
| RTX A4000 CUDA | Fixed square, backend control | 20.77–23.02 s | 14.8–16.4 photos/s | 16.37 ms | 61.64 ms |
| Intel UHD 770 OpenVINO | Fixed square, FP16 | 71.85–72.08 s | 4.72–4.73 photos/s | 171.62 ms | 211.14 ms |

The fixed-square CUDA control is the fair backend comparison because the
OpenVINO export has a fixed 1280×1280 input. Under matched preprocessing, the
UHD 770 is approximately 3.3× slower than the A4000. That difference is
unimportant for FrameFeed's normal background workload: one photo takes about
0.21 seconds, and the entire corpus takes about 72 seconds.

The iGPU was also serving an existing media-transcode workload during the test.
This makes the result representative of the server's real operating conditions,
not an isolated best case.

## Detection agreement

Comparing the fixed-square CUDA control with the OpenVINO iGPU output:

| Measure | Result |
| --- | ---: |
| CUDA person boxes | 743 |
| OpenVINO person boxes | 748 |
| Photos with identical detection counts | 309 / 340 |
| CUDA-box recall at IoU 0.5 | 98.12% |
| OpenVINO-box precision at IoU 0.5 | 97.46% |
| Median IoU for matched boxes | 99.55% |
| Median OpenVINO minus CUDA confidence | -0.000181 |

The same cropper and face cache were then run with both person-box sets. Of 337
photos with matching face analysis:

| Crop comparison | Result |
| --- | ---: |
| Identical crop mode | 332 / 337 (98.52%) |
| Same rendered crop within 2% per coordinate | 334 / 337 (99.11%) |
| One backend contained while the other cropped | 1 / 337 |
| 95th-percentile crop-coordinate difference | 0.082% |

## Precision choice

An FP32 OpenVINO export produced detection records identical to the FP16 export
on all 340 photos, but its corpus pass took 102.34 seconds instead of about 72
seconds. FP16 is therefore the preferred UHD 770 format for this model and
server.

## Recommendation

The Intel UHD 770 is suitable for Advanced CV and can free the A4000 for other
workloads. Use the FP16 OpenVINO export, expose `/dev/dri` to the detector
container, and select OpenVINO device `GPU`.

Keep the Basic face-only crop as the safety fallback. Before changing a
different server, rerun this benchmark against its own corpus because driver
versions, concurrent media workloads, and model exports can affect both speed
and borderline detections.
