# Fold towel π0.5 preparation

- Preserve the completed ACT project and checkpoints. This project is a separate π0.5 fine-tune.
- Never access robot, live camera, CAN or motor endpoints. Only datasets and offline model/GPU tests are in scope.
- Credentials, raw recordings, converted image stores, pretrained weights, fine-tuned weights and environments never enter Git.
- Reuse the ACT complete-episode split and the verified physical leader targets, not future follower feedback.
- YAM joint order: left joints0–5, left gripper, right joints0–5, right gripper. No ALOHA hardware flips or gripper conversion.
- π0.5 target contract: relative 12 joint positions to the observation anchor, absolute normalized grippers. Keep this explicitly separate from ACT all-14 delta semantics.
- Record upstream OpenPI commit and base-checkpoint provenance; do not edit shared upstream code/environments.
- Normalization and reset-pose summary use only the training episodes. Validation metrics are not folding success.
- Default command is preflight/preparation. GPU smoke or training requires explicit CLI mode and an idle GPU; never kill other jobs.

- User explicitly authorized occupied-GPU VRAM reuse for offline evaluation on 2026-11-03. Evaluation may use --share-with-headroom, with an explicit JAX allocator budget plus at least 8GiB spare reserve, prefer lower compute load, and never stop foreign processes. This does not change the separate training launcher defaults.

- User requires all subsequent code and documentation changes to be committed and pushed to k1deler/fold-towel-; verify the remote commit after pushing. Never include credentials, recordings or weights.
