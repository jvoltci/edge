# edge

Static assets for **[jvoltci.github.io/tools](https://jvoltci.github.io/tools)** — the
Whisper speech-recognition weights and the ONNX Runtime wasm builds its Transcribe
tool loads.

There is no code here. Nothing is built. This repo is a bucket that GitHub Pages
happens to serve.

## Why it is a separate repo

The tools app promises that nothing you open in it is ever uploaded. Keeping that
promise for speech recognition means the model has to be fetched from somewhere,
and the two obvious somewheres both fail:

- **A third-party CDN** (jsdelivr, huggingface.co) is a request to someone else's
  server every time the tool starts. The audio would still never leave the
  machine — but "no third party learns you opened this" is part of what the app
  is for, and a CDN request gives that away.
- **The tools repo itself** would carry 158 MB of binaries in its git history
  forever, on every clone, for an app that is otherwise a few hundred kilobytes.

So: a second repo, published to Pages. `jvoltci.github.io/edge/` and
`jvoltci.github.io/tools/` are the **same origin** — GitHub serves every project
site of a user from one host — so there is no CORS boundary, no preflight, and
no third party involved. The weights are one directory over from the app that
loads them.

For the record, the first idea was GitHub Releases, and it does not work. Release
assets are served with **no `access-control-allow-origin` header on either hop** —
neither the `github.com` redirect nor the `release-assets.githubusercontent.com`
response carries one — so a browser `fetch()` cannot read them. That is not a
setting anyone can turn on. Pages, by contrast, sends `access-control-allow-origin: *`,
and here does not even need to.

## What is served

| Path | Bytes | What |
|---|---|---|
| `models/whisper-tiny/` | 45 MB | Whisper tiny, int8. Fast, rougher. |
| `models/whisper-base/` | 81 MB | Whisper base, int8. Slower, noticeably better. |
| `ort/<version>/` | 35 MB | ONNX Runtime wasm, plain + asyncify builds. |

Both models are **multilingual**, not the `.en` variants. This costs nothing:
`whisper-base` and `whisper-base.en` quantise to byte-identical sizes, so the
English-only builds have no advantage to trade for their 98 missing languages.

The runtime directory is **version-pinned in its path** on purpose. It holds the
exact bytes out of the `onnxruntime-web` package the app was compiled against.
Bumping that dependency changes the path, so a stale pairing fails loudly at
fetch time instead of quietly loading a runtime that disagrees with the caller.

## Rebuilding

```sh
./fetch.sh          # re-downloads everything; commit whatever changes
```

The script is the provenance record — every file here came from either
`huggingface.co/onnx-community` or the tools repo's `node_modules`, and it says
which.

## Two limits that shaped this

- **100 MiB** is GitHub's hard block on a single file in a normal push. Whisper
  *small*'s merged decoder is 156 MB, which is why it is not offered here. Git
  LFS is the usual answer and is not available: [GitHub's own docs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
  state that "Git LFS cannot be used with GitHub Pages sites."
- **1 GB** is the published Pages site limit, with a soft 100 GB/month of
  bandwidth. At 158 MB there is plenty of room, but not room for carelessness.

## Licences

The model weights are ONNX exports by [onnx-community](https://huggingface.co/onnx-community)
of OpenAI's Whisper, which is **MIT**. The wasm binaries are
[ONNX Runtime](https://github.com/microsoft/onnxruntime), also **MIT**. Neither
was modified — the files are byte-for-byte what those projects publish.
