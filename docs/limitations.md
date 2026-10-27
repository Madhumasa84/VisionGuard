# Scientific Limitations and Safety Disclosures

VisionGuard Lite is an open-source academic research demonstration and baseline implementation. To maintain scientific integrity, researchers and prospective deployers must consider the following fundamental limitations.

---

## 1. Small Staged-Fall Participant Cohort
- **Participant Diversity**: The primary real-data benchmark (GMDCSA-24) was recorded with **only 4 participants** (healthy young adults).
- **Staged Conditions**: Falls were intentionally enacted onto cushioned mats under controlled laboratory lighting. Authentic real-world falls in geriatric cohorts involve distinct biomechanics (frailty, loss of consciousness, involuntary muscle spasms, attempts to break the fall, and gradual slips).
- **Physical Assistive Devices**: Neither wheelchairs, walkers, canes, nor bed rails were included in the training cohort. Assistive equipment can introduce significant skeleton occlusion and tracking artifacts.

---

## 2. False Alarm Frequency & Operational Trade-Offs
- **Unconfirmed Window-Level False Alarms**: In continuous unconfirmed monitoring ($K=1$), isolated window false alarms occur at **119.72 false alarms per hour** on TemporalCNN (up to **140.43 FA/hr** on BaselineMLP), equivalent to approximately **one false alarm every 25.6–30.1 seconds**.
- **Calibration Limitations**: Temperature scaling ($T=1.627$) only slightly reduces ECE (from 63.3% to 59.3%), indicating that probability confidence values alone cannot prevent false alarms under cross-subject shift.
- **Causes of Transient False Positives**:
  - Rapid transitions (e.g., throwing oneself onto a sofa or bed).
  - Vigorous floor-level exercises (stretching, yoga, push-ups).
  - Reaching rapidly down to pick up dropped items.
- **Temporal Confirmation Trade-Off**: Requiring $K=2$ consecutive positive windows suppresses false alarms to **95.00 FA/hr** while maintaining **92.2% event recall**, but introduces a **2.18-second mean physical detection delay in the historical evaluation**. Requiring $K=4$ reduces false alarms to 83.80 FA/hr at the cost of missing 29.9% of falls (70.1% recall) and a 3.14-second delay.

---

## 3. Potential for Missed Falls (False Negatives)
- **Occlusions**: Obscured landmarks, such as a person falling behind furniture, can make pose estimation unreliable. The demo rejects low-visibility confirmation windows, but this does not guarantee that tracking failures will always be detected.
- **Gradual Slumps**: Slow, assisted slips down a wall or furniture may exhibit low vertical velocity, which can fail to cross the temporal convolution activation threshold.

---

## 4. Hardware & Benchmark Distinction
- **Server vs Consumer CPU**: Colab benchmarks running on multi-core cloud instances (Intel Xeon / AMD EPYC) or server GPUs do not reflect consumer laptop constraints.
- **Thermal & Battery Throttling**: On portable battery-powered laptops or edge boards (Raspberry Pi, low-power NUCs), thermal throttling and background OS tasks can increase MediaPipe frame processing latency from 23 ms to > 60 ms.
- **Mobile Claims**: Desktop CPU benchmarks do not constitute verification on mobile architectures (ARM NEON / Android NPU). Mobile deployment requires separate integration and on-device validation; this repository does not provide a validated mobile application.

---

## 5. Regulatory & Life-Safety Disclaimers

> [!CAUTION]
> **NOT A MEDICAL DEVICE OR LIFE-SAFETY PRODUCT**:
> - VisionGuard Lite is **not** certified by the FDA, CE, or any regulatory body as a medical diagnostic or emergency life-safety device.
> - **Never** rely on this software as the sole mechanism for elderly care, critical patient monitoring, or automated emergency services dispatch.
> - The software is provided "as is", without warranty of any kind, strictly for academic and research evaluation.


## 6. Demo scope and retention

The interface analyzes bounded clips, targeting 10 FPS and stopping at 250 sampled frames. Actual sampling rate depends on source FPS, and repeated-pose padding is used for short clips. These behaviors can change the temporal context seen by the model.

Low mean pose visibility forces an uncertain clip summary. A confirmation only means the configured model policy was met, not that a physical fall or injury was independently verified. Classifier probabilities can remain high even when no person is tracked.

Uploaded video and generated files are stored on the app host. Per-run pipeline artifacts use the system temporary directory and do not have an application-enforced secure-deletion policy. Do not expose the default unauthenticated app to untrusted users. See [architecture](architecture.md) and [setup](setup.md).
