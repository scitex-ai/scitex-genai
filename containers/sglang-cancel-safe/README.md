# Cancellation-safe pinned SGLang image

This directory builds a new, immutable Apptainer image from the exact SGLang
OCI image used during the 2026-09-11 FigRecipe incident. It does not alter an
existing SIF or a running service.

## What is pinned

- OCI base: `lmsysorg/sglang@sha256:45e39d4c5bcfd89d171b3358ba78899354ab26a85bc746a17621ad818f8394aa`
- SGLang source: `4ccff141dbe992794f9da6c3aa23535b4f72000d`
- Vendored patch SHA-256: `36d9ea8c3b386608e00967b881e96115c3ee5d0ec423857315e55ef1b4c7bfeb`

The image build verifies the source commit before applying the patch and fails
if the patch no longer applies. The changes are adapted to this pinned branch
from upstream commit `f478b2bb2d582c09e7f1b4e49f0c2d039da8747a`.

The two patch components address the observed failure modes:

1. A disconnected HTTP handler no longer deletes a request state that was
   already dispatched. It sends one idempotent scheduler abort and retains the
   state until the scheduler acknowledges it. This is the failure described in
   [SGLang #36333](https://github.com/sgl-project/sglang/issues/36333).
2. A deferred chunked-prefill abort is retried if the request moves out of the
   chunked slot during overlap scheduling. This closes the transition window
   described in [SGLang #36876](https://github.com/sgl-project/sglang/issues/36876).

## Build

Build on a compute node with scratch space. From an existing Slurm allocation,
use an overlapping step; do not run the image build on a login node:

```bash
srun --overlap --ntasks=1 --cpus-per-task=8 \
  containers/sglang-cancel-safe/build.sh \
  /scratch/$USER/images/sglang-qwen38-cancel-safe-4ccff141.sif
```

The builder refuses paths outside `/scratch`, refuses to overwrite an image,
uses a build-local cache, runs the definition's tests, and checks the embedded
labels. Keep the completed SIF under a content-specific filename.

## GPU acceptance

The acceptance launcher is intentionally restricted to a Slurm allocation. It
starts an isolated server on port 18792 with TP=2, 1M context, 32K chunked
prefill, overlap scheduling (the default), EAGLE, and metrics. It sends a unique
2.4 MB cold prompt, waits until the request is visible in `/v1/loads`, closes
the client socket during prefill, and requires the server to return to zero
running and waiting requests. It also rejects the incident's
`state was deleted` signature for the test request ID.

From a dedicated two-GPU allocation:

```bash
srun --overlap --ntasks=1 --gres=gpu:2 \
  containers/sglang-cancel-safe/run_acceptance.sh \
  /scratch/$USER/images/sglang-qwen38-cancel-safe-4ccff141.sif \
  /path/to/the/qwen-model
```

Acceptance artifacts and the server log are retained under
`/scratch/sglang-cancel-acceptance.*`. This test must target only the isolated
server it launches, never the production tunnel or gateway port.

## Promotion and rollback

Promotion is a separate, controlled operation after review and GPU acceptance:

1. Record the new SIF's `sha256sum`, `apptainer inspect --labels` output, GPU
   test log, and previous configured SIF path.
2. Change the model service configuration to the new unique SIF path during a
   maintenance window. Do not replace the old SIF in place.
3. Verify health, a short completion, disconnect drain, and gateway load before
   restoring normal traffic.

To roll back, restore the recorded previous SIF path in the model service
configuration and restart only the controlled replacement allocation/service.
The rollback does not require rebuilding or deleting either image. If a request
is in flight, drain or explicitly abort it before the controlled restart; do not
reuse the production port for acceptance testing.

