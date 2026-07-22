# Hedron

Hedron is a GPU inference server for running Hugging Face and local models behind an OpenAI-compatible HTTP API. It serves text generation, multimodal chat, embeddings, reranking, image generation, speech, and transcription from one container interface.

This README is the public operations manual. You do not need the source repository, a Python package, or a TOML file to deploy Hedron.

[Quick start](#quick-start) · [Choose an image](#choose-an-image) · [Load a model](#load-a-model) · [Call the API](#call-the-api) · [Operate Hedron](#operate-hedron) · [Troubleshoot](#troubleshoot)

## Quick start

You need:

- an NVIDIA GPU supported by one of the image targets below;
- a host driver capable of running CUDA 13 containers;
- Docker configured with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Start a Qwen3.5 GGUF model. The default image detects the visible GPU architecture at startup:

```bash
docker run --name hedron --rm --gpus all \
  -p 127.0.0.1:8080:80 \
  -v "${HOME}/.cache/huggingface:/data" \
  ghcr.io/geodesa-ai/hedron:latest \
  --port 80 \
  --model hf://unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf
```

The first start downloads the model. Follow startup and model-loading progress with:

```bash
docker logs -f hedron
```

In another terminal, wait for the server and inspect the loaded model:

```bash
until curl --fail --silent http://127.0.0.1:8080/health >/dev/null; do sleep 2; done
curl --fail http://127.0.0.1:8080/v1/models
```

Then send a request:

```bash
curl --fail http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Explain speculative decoding in one paragraph."}
    ],
    "max_tokens": 256
  }'
```

`default` selects the container's default loaded model. Use the exact ID returned by `/v1/models` when addressing a specific model in a multi-model deployment.

## Choose an image

Images are published at [`ghcr.io/geodesa-ai/hedron`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron). Every image contains all supported model architectures and the production kernel set; tags select GPU machine code and CUDA runtime, not model families. Experimental quantization kernels are opt-in source-build features and are not carried in release images. SM121 and automatic tags are multi-platform manifests with native `linux/amd64` and `linux/arm64` variants, so Docker selects the correct executable for GB10/DGX Spark automatically. Other architecture-specific tags currently target `linux/amd64`.

| CUDA target | GPU generation and examples | Host platform | CUDA 12 rolling tag | CUDA 13 rolling tag |
|---|---|---|---|---|
| `sm_75` | Turing: T4, RTX 20xx | `linux/amd64` | [`sm75-cu12`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm75-cu12) | [`sm75-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm75-cu13) |
| `sm_80` | Ampere: A100, A30 | `linux/amd64` | [`sm80-cu12`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm80-cu12) | [`sm80-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm80-cu13) |
| `sm_86` | Ampere: RTX 30xx, A10, A40 | `linux/amd64` | [`sm86-cu12`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm86-cu12) | [`sm86-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm86-cu13) |
| `sm_87` | Ampere: Jetson Orin GPU target | `linux/amd64` | [`sm87-cu12`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm87-cu12) | [`sm87-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm87-cu13) |
| `sm_89` | Ada: L4, L40/L40S, RTX 40xx | `linux/amd64` | [`sm89-cu12`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm89-cu12) | [`sm89-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm89-cu13) |
| `sm_90a` | Hopper: H100, H200, GH200 | `linux/amd64` | [`sm90a-cu12`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm90a-cu12) | [`sm90a-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm90a-cu13) |
| `sm_100f` | Blackwell: B200, GB200 | `linux/amd64` | [`sm100f-cu12`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm100f-cu12) | [`sm100f-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm100f-cu13) |
| `sm_103f` | Blackwell Ultra: B300, GB300 | `linux/amd64` | [`sm103f-cu12`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm103f-cu12) | [`sm103f-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm103f-cu13) |
| `sm_110f` | Blackwell: Jetson T5000/T4000 GPU target | `linux/amd64` | — | [`sm110f-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm110f-cu13) |
| `sm_120f` | Blackwell: RTX 50xx, RTX PRO Blackwell | `linux/amd64` | [`sm120f-cu12`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm120f-cu12) | [`sm120f-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm120f-cu13) |
| `sm_121f` | Blackwell: GB10, DGX Spark | `linux/amd64`, `linux/arm64` | [`sm121f-cu12`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm121f-cu12) | [`sm121f-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm121f-cu13) |
| automatic | Selects the visible GPU at startup | `linux/amd64`, `linux/arm64` | [`cu12`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=cu12) | [`latest`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=latest), [`cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=cu13) |

Architecture-specific images are smaller and are the recommended production choice. Choose CUDA 12 when the host driver cannot load a CUDA 13 runtime; the CUDA toolkit installed on the host does not affect containers. CUDA 12.9 predates SM110, so that target is CUDA 13 only. On `linux/amd64`, automatic images package every target available for their CUDA line. On `linux/arm64`, they currently package SM121 for GB10. Both select the visible GPU using `nvidia-smi`. The SM87 and SM110 rows identify GPU targets but their current x86 images do not run on Jetson's ARM host. If GPUs with different compute capabilities are visible in one container, set an explicit target such as `-e HEDRON_CUDA_TARGET=sm89` or expose only one GPU generation.

Rolling tags move when a new Hedron version is published. `latest` and the bare immutable release tag, such as `v0.7.1`, select the CUDA 13 automatic image. Explicit automatic tags are `cu12`, `cu13`, `v0.7.1-cu12`, and `v0.7.1-cu13`. Architecture-specific releases place the version first, as in `v0.7.1-sm89-cu13`. Prefer an immutable tag and record its digest for reproducible deployments.

## Load a model

The container entrypoint is `hedron-server`. Its command shape is:

```text
hedron-server [SERVER OPTIONS] --model SOURCE [MODEL SETTINGS] [--model SOURCE [MODEL SETTINGS] ...]
```

There are no loader subcommands and no JSON or TOML model selector. Hedron infers the loader from model metadata and the artifact container. A server requires `--port`, `--mcp-port`, or `--interactive-mode`.

Safetensors repositories and directories are capability-detected for text, multimodal, embeddings, reranking, Stable Diffusion 1.5/XL/3 image generation, speech, and transcription. Flux and LTX manifests are detected and fail with a specific startup error, but those pipelines still require external component-source configuration that the unified container CLI does not expose; they are not operable through this release interface. A direct `.gguf` locator is intentionally a text-generation container; non-text GGUF, projector/component layouts, and GGUF embeddings or audio are not supported through the unified interface yet. Use a supported safetensors repository or directory for those capabilities rather than relying on filename guessing.

Every source uses an explicit scheme so a repository name can never be mistaken for a filesystem path:

| Source | Meaning | Example |
|---|---|---|
| `hf://` | Hugging Face repository, optionally followed by one GGUF artifact | `hf://Qwen/Qwen3.5-9B` or `hf://org/repo/model.gguf` |
| `file:` | Relative or absolute local/mounted path | `file:./model` or `file:///models/model.gguf` |
| `s3://` | S3-compatible object or prefix, staged into the local model cache | `s3://models/releases/qwen` |
| `nfs://` | Path in an already-mounted NFS export | `nfs://storage/models/qwen` |

Settings after a `--model` belong to that model until the next `--model`. A setting may appear only once in each declaration. The first declaration is the default model:

```bash
hedron-server --port 80 \
  --model hf://org/primary --revision 4d2c1f0 --isq Q4K \
  --model file:///models/embedding --dtype bf16
```

Common model-scoped settings include `--revision`, `--dtype`, `--isq`, `--topology`, `--num-device-layers`, `--chat-template`, `--jinja-explicit`, `--expert-gpu-slots`, and the speculative-decoding settings below. Put each setting after the model it configures.

Inspect the exact options shipped in an image:

```bash
docker run --rm ghcr.io/geodesa-ai/hedron:latest --help
```

### Hugging Face authentication

For a gated or private repository, pass `HF_TOKEN` into the container. Hedron detects it automatically:

```bash
docker run --name hedron --rm --gpus all \
  -e HF_TOKEN \
  -p 127.0.0.1:8080:80 \
  -v "${HOME}/.cache/huggingface:/data" \
  ghcr.io/geodesa-ai/hedron:latest \
  --port 80 --model hf://OWNER/GATED-MODEL
```

By default Hedron checks `HF_TOKEN`, then `HUGGING_FACE_HUB_TOKEN`, then the cached Hugging Face token. `--token-source` is available only when an operator needs to override that behavior with `auto`, `env:NAME`, `path:FILE`, `cache`, `literal:TOKEN`, or `none`. Avoid placing literal tokens in shell history.

### Local models

Mount a local directory read-only and pass the container path as the model ID:

```bash
docker run --name hedron --rm --gpus all \
  -p 127.0.0.1:8080:80 \
  -v /srv/models:/models:ro \
  ghcr.io/geodesa-ai/hedron:latest \
  --port 80 --model file:///models/qwen/model.gguf
```

For S3-compatible storage, provide the normal AWS credential and endpoint environment variables. Hedron streams objects into `HEDRON_MODEL_CACHE` (or the platform cache directory) before loading. For NFS, mount the export into the container and set `HEDRON_NFS_ROOT` to that mount, or configure the conventional `/net/<host>/...` autofs map; Hedron does not perform privileged mounts itself.

## Call the API

The interactive API reference is available from a running server at `http://127.0.0.1:8080/docs`; the OpenAPI document is at `/api-doc/openapi.json`.

| Capability | Endpoint |
|---|---|
| Liveness | `GET /health` |
| Loaded models | `GET /v1/models` |
| Chat and multimodal chat | `POST /v1/chat/completions` |
| Text completions | `POST /v1/completions` |
| Responses API | `/v1/responses` |
| Embeddings | `POST /v1/embeddings` |
| Reranking | `POST /v1/rerank` |
| Image generation | `POST /v1/images/generations` |
| Video API surface (no operable auto-loaded pipeline in this release) | `/v1/videos`, `/v1/videos/generations` |
| Speech generation | `POST /v1/audio/speech` |
| Transcription | `POST /v1/audio/transcriptions` |
| Prometheus metrics | `GET /metrics` |

Only endpoints supported by the loaded model are useful. For example, an embedding model serves embeddings rather than chat completions.

Hedron can also expose an MCP server on a separate port. Set `--mcp-port PORT` and connect to `/mcp` on that port.

## Tune a deployment

Common server-level controls apply to the whole process:

| Option | Effect |
|---|---|
| `--max-seqs N` | Maximum concurrently running sequences |
| `--pa-ctxt-len N` | Size PagedAttention for a target context length |
| `--pa-gpu-mem-usage FRACTION` | Fraction of GPU memory available to PagedAttention |
| `--pa-prefill-chunk-size N` | Bound prefill chunks to reduce peak memory |
| `--pa-cache-type auto\|f8e4m3\|tq3` | KV-cache storage type |
| `--rate-limit-requests N` | Enable a per-identity, in-process fixed-window request limit |
| `--rate-limit-window-secs N` | Set that fixed window; defaults to 60 seconds |

`--num-device-layers` and `--isq` are model-scoped; put them after the relevant `--model` declaration.

The complete CLI in the image is authoritative. Start with defaults, measure the workload, and change one resource constraint at a time.

### Speculative decoding

Speculative decoding is a first-class model setting, not a per-request option.

MTP uses the prediction head embedded in the target model:

```bash
docker run --rm --gpus all -p 127.0.0.1:8080:80 \
  ghcr.io/geodesa-ai/hedron:latest \
  --port 80 \
  --model hf://OWNER/TARGET-GGUF/TARGET.gguf \
  --speculator-mode mtp --speculator-max-draft-tokens 4
```

DFlash loads a draft head from a Hugging Face repository or a local snapshot containing `config.json` and `model.safetensors`:

```bash
docker run --rm --gpus all -p 127.0.0.1:8080:80 \
  ghcr.io/geodesa-ai/hedron:latest \
  --port 80 \
  --model hf://OWNER/TARGET-GGUF/TARGET.gguf \
  --speculator-mode dflash \
  --speculator-source hf://OWNER/DFLASH-HEAD \
  --speculator-max-draft-tokens 8
```

Both modes apply to every generation request. Startup fails if Hedron cannot attach the requested head; it does not silently fall back to ordinary autoregressive decoding.

### Expert offloading

For GGUF MoE models, bound the routed experts resident on the GPU and, optionally, the pinned-host warm tier:

```bash
docker run --rm --gpus all -p 127.0.0.1:8080:80 \
  ghcr.io/geodesa-ai/hedron:latest \
  --port 80 \
  --model hf://OWNER/MOE-GGUF/MODEL.gguf \
  --expert-gpu-slots 24 --expert-host-slots 64
```

GPU slots must cover the model's routed top-k. Find `k` in the Hugging Face `config.json` as `num_experts_per_tok`, or in GGUF metadata as `<architecture>.expert_used_count`. Start with `ceil(1.5 × k)` slots and increase it as GPU memory permits. Hedron reports the detected `k` and a recommendation during startup. Setting exactly `k` is valid but leaves no spare residency and is likely to thrash as routing changes between tokens.

`--expert-host-slots` requires `--expert-gpu-slots`. Omit the host value to let Hedron size the warm tier from available host memory.

## Operate Hedron

For a long-running deployment, use a fixed architecture tag, persist `/data`, name the container, and configure restart behavior:

```bash
docker run -d --name hedron --restart unless-stopped --gpus all \
  -p 127.0.0.1:8080:80 \
  -v "${HOME}/.cache/huggingface:/data" \
  ghcr.io/geodesa-ai/hedron:v0.7.0-sm89-cu13 \
  --port 80 \
  --model hf://unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf
```

Useful operating commands:

```bash
docker logs -f hedron
docker inspect --format '{{.State.Status}}' hedron
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/v1/models
curl --fail http://127.0.0.1:8080/metrics
docker stop --time 30 hedron
```

The server writes logs to stdout and stderr, exits nonzero when configuration or model loading fails, and receives normal container stop signals directly. The HTTP listener becomes available after model initialization, so `/health` and `/v1/models` are also practical readiness checks. Persisting `/data` avoids downloading weights again after replacement or restart.

### Security

Hedron does not provide transport encryption or validate API keys. Its optional rate limiter is local to one server process and uses an API-key header or peer IP only as the bucket identity; it is not authentication or a distributed quota system. Do not expose Hedron directly to an untrusted network. Keep the published port on localhost or a private network, or place it behind a reverse proxy or gateway that provides TLS, authentication, distributed request limits, and audit policy.

## Troubleshoot

| Symptom | What to check |
|---|---|
| Docker cannot see a GPU | Run `docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi`; repair the host driver or NVIDIA Container Toolkit first. |
| `no kernel image is available` | The architecture tag does not match the GPU. Use the matching tag or `latest`. |
| `latest` cannot select a binary | Ensure `nvidia-smi` is available in the container and only one GPU generation is visible, or set `HEDRON_CUDA_TARGET`. |
| Hugging Face returns 401 or 403 | Pass the token with `-e HF_TOKEN`, confirm the account has accepted any model license, and verify the repository name. |
| A request says the model does not exist | Use `"model":"default"` or copy the exact ID from `/v1/models`. |
| CUDA out of memory | Choose a smaller quantization, lower `--max-seqs` or `--pa-ctxt-len`, bound prefill with `--pa-prefill-chunk-size`, or configure layer/expert offloading. |
| Startup appears idle | Model weights or kernels may still be loading. Check `docker logs -f hedron`, network access, free disk space, and the persisted cache. |
| A CLI option is rejected | Confirm the source has an explicit scheme and each model-specific option follows the `--model` it configures; inspect `--help`. |

## Report a problem

Use the public [Hedron support and bug-report discussions](https://github.com/geodesa-ai/hedron-community/discussions). Include enough information to reproduce the behavior:

- the full image tag and digest;
- GPU name, driver, and compute capability from `nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv`;
- the full container command with credentials and private paths removed;
- the model repository, revision, and artifact filename;
- the first relevant startup or request logs;
- the request payload with secrets or private data removed;
- the output of `/v1/models` when the server starts successfully.

## License

MIT.
