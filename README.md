# Hedron

Hedron is a GPU inference server for running Hugging Face and local models behind an OpenAI-compatible HTTP API. It serves text generation, multimodal chat, embeddings, reranking, image and video generation, speech, and transcription from one container interface.

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
  gguf -m unsloth/Qwen3.5-9B-GGUF -f Qwen3.5-9B-Q4_K_M.gguf
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

Images are published at [`ghcr.io/geodesa-ai/hedron`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron). Every image contains the full supported model and kernel set; tags select GPU machine code, not model families. Images currently use a CUDA 13 runtime and target `linux/amd64`.

| CUDA target | GPU generation and examples | Rolling tag |
|---|---|---|
| `sm_75` | Turing: T4, RTX 20xx | [`sm75-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm75-cu13) |
| `sm_80` | Ampere: A100, A30; compatible with RTX 30xx | [`sm80-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm80-cu13) |
| `sm_89` | Ada: L4, L40/L40S, RTX 40xx | [`sm89-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm89-cu13) |
| `sm_90a` | Hopper: H100, H200, GH200 | [`sm90a-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm90a-cu13) |
| `sm_100f` | Blackwell: B200, GB200 | [`sm100f-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm100f-cu13) |
| `sm_103f` | Blackwell Ultra: B300, GB300 | [`sm103f-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm103f-cu13) |
| `sm_120f` | Blackwell: RTX 50xx, RTX PRO Blackwell | [`sm120f-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm120f-cu13) |
| `sm_121f` | Blackwell: GB10, DGX Spark | [`sm121f-cu13`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=sm121f-cu13) |
| automatic | All targets above; selects at startup | [`latest`](https://github.com/geodesa-ai/hedron/pkgs/container/hedron?tag=latest) |

Architecture-specific images are smaller and are the recommended production choice. The default image packages every architecture-specific server binary and selects one using `nvidia-smi`. If GPUs with different compute capabilities are visible in one container, set an explicit target such as `-e HEDRON_CUDA_TARGET=sm89` or expose only one GPU generation.

Rolling tags move when a new Hedron version is published. The automatic image uses `latest`; its immutable counterpart is the bare release tag, such as `v0.7.0`. Architecture-specific releases use tags such as `sm89-cu13-v0.7.0`. Prefer an immutable tag and record its digest for reproducible deployments.

## Load a model

The container entrypoint is `hedron-server`. Its command shape is:

```text
hedron-server [SERVER OPTIONS] <MODEL COMMAND> [MODEL OPTIONS]
```

Server options must appear before the model command. Model options must appear after it. A server requires `--port`, `--mcp-port`, or `--interactive-mode`.

These are the model commands most operators need:

| Command | Use it for | Minimum model argument |
|---|---|---|
| `run` | Automatically detected text or supported vision safetensors | `-m OWNER/MODEL` |
| `plain` | Explicit text safetensors loading | `-m OWNER/MODEL` |
| `gguf` | Text GGUF | `-m OWNER/REPO -f MODEL.gguf` |
| `vision-plain` | Vision safetensors | `-m OWNER/MODEL` |
| `vision-gguf` | Vision GGUF model and projector | `-m OWNER/REPO -f MODEL.gguf -f MMPROJ.gguf` |
| `embedding` | Safetensors embedding model | `-m OWNER/MODEL` |
| `gguf-embedding` | GGUF embedding model | `-m OWNER/REPO -f MODEL.gguf` |
| `rerank` | Cross-encoder reranker | `-m OWNER/MODEL` |
| `diffusion` | Image or video generation model | `-m OWNER/MODEL` |
| `speech` | Text-to-speech model | `-m OWNER/MODEL` |
| `transcription` | Speech-to-text model | `-m OWNER/MODEL` |
| `multi-model` | Models declared in a JSON configuration | `-c /path/config.json` |

`run` is not the GGUF loader. Use `gguf` whenever the artifact is a `.gguf` file.

Inspect the exact options shipped in an image:

```bash
docker run --rm ghcr.io/geodesa-ai/hedron:latest --help
docker run --rm ghcr.io/geodesa-ai/hedron:latest gguf --help
docker run --rm ghcr.io/geodesa-ai/hedron:latest diffusion --help
```

### Hugging Face authentication

For a gated or private repository, pass `HF_TOKEN` into the container. Hedron detects it automatically:

```bash
docker run --name hedron --rm --gpus all \
  -e HF_TOKEN \
  -p 127.0.0.1:8080:80 \
  -v "${HOME}/.cache/huggingface:/data" \
  ghcr.io/geodesa-ai/hedron:latest \
  --port 80 run -m OWNER/GATED-MODEL
```

By default Hedron checks `HF_TOKEN`, then `HUGGING_FACE_HUB_TOKEN`, then the cached Hugging Face token. `--token-source` is available only when an operator needs to override that behavior with `auto`, `env:NAME`, `path:FILE`, `cache`, `literal:TOKEN`, or `none`. Avoid placing literal tokens in shell history.

### Local models

Mount a local directory read-only and pass the container path as the model ID:

```bash
docker run --name hedron --rm --gpus all \
  -p 127.0.0.1:8080:80 \
  -v /srv/models:/models:ro \
  ghcr.io/geodesa-ai/hedron:latest \
  --port 80 gguf -m /models/qwen -f model.gguf
```

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
| Video generation and management | `/v1/videos`, `/v1/videos/generations` |
| Speech generation | `POST /v1/audio/speech` |
| Transcription | `POST /v1/audio/transcriptions` |
| Prometheus metrics | `GET /metrics` |

Only endpoints supported by the loaded model are useful. For example, an embedding model serves embeddings rather than chat completions.

Hedron can also expose an MCP server on a separate port. Place `--mcp-port PORT` before the model command and connect to `/mcp` on that port.

## Tune a deployment

Common server-level controls are placed before the model command:

| Option | Effect |
|---|---|
| `--max-seqs N` | Maximum concurrently running sequences |
| `--pa-ctxt-len N` | Size PagedAttention for a target context length |
| `--pa-gpu-mem-usage FRACTION` | Fraction of GPU memory available to PagedAttention |
| `--pa-prefill-chunk-size N` | Bound prefill chunks to reduce peak memory |
| `--pa-cache-type auto\|f8e4m3\|f8e5m2` | KV-cache storage type |
| `--num-device-layers ...` | Explicit layer placement across devices |
| `--isq TYPE` | In-situ quantization for supported safetensors models |

The complete CLI in the image is authoritative. Start with defaults, measure the workload, and change one resource constraint at a time.

### Speculative decoding

Speculative decoding is a deployment setting, not a per-request option. It is currently attached through the text GGUF loader.

MTP uses the prediction head embedded in the target model:

```bash
docker run --rm --gpus all -p 127.0.0.1:8080:80 \
  ghcr.io/geodesa-ai/hedron:latest \
  --port 80 \
  --speculator-mode mtp --speculator-max-draft-tokens 4 \
  gguf -m OWNER/TARGET-GGUF -f TARGET.gguf
```

DFlash loads a draft head from a Hugging Face repository or a local snapshot containing `config.json` and `model.safetensors`:

```bash
docker run --rm --gpus all -p 127.0.0.1:8080:80 \
  ghcr.io/geodesa-ai/hedron:latest \
  --port 80 \
  --speculator-mode dflash \
  --speculator-source OWNER/DFLASH-HEAD \
  --speculator-max-draft-tokens 8 \
  gguf -m OWNER/TARGET-GGUF -f TARGET.gguf
```

Both modes apply to every generation request. Startup fails if Hedron cannot attach the requested head; it does not silently fall back to ordinary autoregressive decoding.

### Expert offloading

For GGUF MoE models, bound the routed experts resident on the GPU and, optionally, the pinned-host warm tier:

```bash
docker run --rm --gpus all -p 127.0.0.1:8080:80 \
  ghcr.io/geodesa-ai/hedron:latest \
  --port 80 \
  --expert-gpu-slots 24 --expert-host-slots 64 \
  gguf -m OWNER/MOE-GGUF -f MODEL.gguf
```

GPU slots must cover the model's routed top-k. Find `k` in the Hugging Face `config.json` as `num_experts_per_tok`, or in GGUF metadata as `<architecture>.expert_used_count`. Start with `ceil(1.5 × k)` slots and increase it as GPU memory permits. Hedron reports the detected `k` and a recommendation during startup. Setting exactly `k` is valid but leaves no spare residency and is likely to thrash as routing changes between tokens.

`--expert-host-slots` requires `--expert-gpu-slots`. Omit the host value to let Hedron size the warm tier from available host memory.

## Operate Hedron

For a long-running deployment, use a fixed architecture tag, persist `/data`, name the container, and configure restart behavior:

```bash
docker run -d --name hedron --restart unless-stopped --gpus all \
  -p 127.0.0.1:8080:80 \
  -v "${HOME}/.cache/huggingface:/data" \
  ghcr.io/geodesa-ai/hedron:sm89-cu13-v0.7.0 \
  --port 80 \
  gguf -m unsloth/Qwen3.5-9B-GGUF -f Qwen3.5-9B-Q4_K_M.gguf
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

Hedron does not provide transport encryption or validate API keys. Do not expose it directly to an untrusted network. Keep the published port on localhost or a private network, or place the server behind a reverse proxy or gateway that provides TLS, authentication, request limits, and audit policy.

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
| A CLI option is rejected | Confirm global options are before the model command and model-specific options are after it; inspect both `--help` levels. |

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
