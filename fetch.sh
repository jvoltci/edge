#!/usr/bin/env bash
#
# Rebuild everything this repo serves, from scratch.
#
# Nothing here is hand-placed. If a file is in this repo it came from one of the
# two sources below, and this script is the record of which. Re-run it after
# bumping ONNX_VERSION or the model list and commit whatever changes.
#
#   models/    <- huggingface.co/onnx-community, the ONNX exports of Whisper
#   ort/       <- the onnxruntime-web npm package, copied out of the tools repo's
#                 node_modules so the bytes served are exactly the bytes the app
#                 was built against
#
# Why only two files per model: onnx-community publishes ~20 quantisation
# variants each. We serve the int8 ("quantized") encoder and the MERGED int8
# decoder. Merged matters — the alternative is shipping decoder_model and
# decoder_with_past_model separately, which is ~2x the bytes for the same result
# because the merged graph handles both the first token and the cached rest.
#
# Why multilingual and not the .en variants: identical byte size. whisper-base
# and whisper-base.en both quantise to 23,201,3xx + 53,69x,xxx. The multilingual
# model handles 99 languages for the same download, so the .en builds have
# nothing to offer.
#
# ── the 100 MiB ceiling, and why it is no longer a ceiling ───────────────────
#
# GitHub hard-blocks any file over 100 MiB on push. Git LFS is the documented
# answer and does not work here, because GitHub Pages serves the LFS pointer
# text rather than the bytes. That limit — not the browser, not the decoder —
# is the whole reason this repo stopped at whisper-base, whose 58% word recall
# was in turn the reason live subtitles looked broken.
#
# So files over the limit are now split here and rejoined in the browser's fetch
# layer (see model-parts.ts in the tools repo). Each `<file>.onnx` over the
# threshold becomes `<file>.onnx.part0`, `.part1`, … and an entry in
# models/parts.json giving the part count and the rejoined length. The app asks
# for the original name and never learns the difference.
#
# Nothing about a model needs to change to be published this way, so the list
# below is now bounded by what runs fast enough, not by what fits.

set -euo pipefail
cd "$(dirname "$0")"

# Pinned to 1.24.3 by an npm override in the tools repo, NOT the version
# transformers.js depends on. ONNX Runtime 1.25 added a graph optimisation that
# rewrites QDQ into MatMulNBits and then rejects these Whisper exports with
# "Missing required scale: model.decoder.embed_tokens.weight_merged_0_scale".
# See microsoft/onnxruntime#28306 and huggingface/transformers.js#1707. 1.24.3
# is the last stable release before that change.
ONNX_VERSION="1.24.3"

# The override nests the package, so look in both places rather than assuming.
for c in \
  "../tools/node_modules/onnxruntime-web/dist" \
  "../tools/node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist"
do
  [ -d "$c" ] && ORT_SRC="$c"
done
: "${ORT_SRC:?onnxruntime-web not found in the tools repo — run npm install there first}"

# `<hf-org>/<hf-name>` — served under <hf-name>. Two orgs now, because the
# distil-whisper models are published by their own authors and not mirrored into
# onnx-community.
#
# whisper-small is deliberately NOT here. It was published, measured against two
# films with their own subtitle tracks, and lost:
#
#                  film 1 WER   film 2      speed        download
#   whisper-base       55.8%    40.9%      3.8x realtime    81 MB
#   whisper-small      57.1%       —       1.2x realtime   253 MB
#
# Three times slower for no accuracy, so it is not worth 240 MB of this repo.
# The splitting machinery it was added to prove out is what stayed.
#
# The distil models are the reason that machinery was worth building. They are
# not "whisper but bigger" — they keep the full encoder and throw away all but
# two decoder layers, so distil-small.en carries a 76 MB decoder where
# whisper-small needs 149 MB. English only, which is why asr.ts has to know
# which models can be told to translate.
# distil-medium.en was published, measured and removed in the same session. It
# is the only model tried so far that produced degenerate output — 17 of its 171
# cues on a seven-minute episode contained no letter at all, just ".", ".." and
# "......", each with its own timestamp, marching across a scene of music:
#
#                    WER     speed
#   distil-small.en  54.4%   4.1x realtime   176 MB
#   whisper-base     55.8%   9.4x realtime    81 MB
#   distil-medium.en 83.6%   1.5x realtime   407 MB
#
# That failure is a decoding problem rather than a model one — OpenAI's own
# implementation re-runs a window at a higher temperature when it detects a
# collapse like this, and nothing here does. Worth revisiting if that gets
# built. Not worth 407 MB in the meantime.
MODELS=(
  onnx-community/whisper-tiny
  onnx-community/whisper-base
  distil-whisper/distil-small.en
  Xenova/modnet
  # Diarization — "who said what" — is two models, not one, and they are both
  # tiny. Segmentation says WHERE someone is speaking; the speaker embedder says
  # WHO, by turning a stretch of speech into a 256-dimension vector that can be
  # clustered. Neither is gated and both were run frame by frame before being
  # published here.
  #
  #   pyannote-segmentation-3.0  MIT. int8 1,542,304 B. Takes 10 s of 16 kHz mono
  #                              as [1,1,160000] and returns [1,589,7] — 7
  #                              powerset classes over 3 local speakers at 58.9
  #                              frames a second, so overlapping speech is a
  #                              class of its own rather than a coin toss.
  #   wespeaker-resnet34-LM      CC-BY-4.0, so it needs attribution, not a
  #                              licence fee. int8 6,685,123 B. Takes an 80-bin
  #                              Kaldi fbank as [1,T,80] and returns [1,256].
  #
  # Measured here before publishing, on 20 s of two synthesised speakers taking
  # five turns: 2 speakers found, and every turn assigned to the right one
  # (cosine 0.76-0.94 within a speaker, 0.02-0.09 across).
  onnx-community/pyannote-segmentation-3.0
  onnx-community/wespeaker-voxceleb-resnet34-LM
)

# Split anything at or above this, in MiB.
#
# 90 leaves ten MiB of headroom under the 100 MiB refusal without splitting
# anything git would have accepted. The tempting alternative is 48, which also
# clears GitHub's 50 MiB *advisory* warning — but whisper-base's decoder is
# 51 MiB, so that would re-cut a file that has been served whole for months and
# invalidate it in every browser cache that already holds it. A one-time warning
# on push is cheaper than a re-download for everyone who has used the tool.
PART_MIB=90

# Everything a tokenizer/processor might reach for. These are all small (the
# largest, tokenizer.json, is ~2.4 MB) so the whole set is mirrored rather than
# guessed at — a missing tokenizer file surfaces as a confusing runtime error
# much later, and the bytes saved by trimming this list are not worth that.
SUPPORT=(
  config.json
  generation_config.json
  preprocessor_config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  added_tokens.json
  vocab.json
  merges.txt
  normalizer.json
)

WEIGHTS=(
  onnx/encoder_model_quantized.onnx
  onnx/decoder_model_merged_quantized.onnx
)

# Which files a given model actually has.
#
# The two arrays above describe whisper's shape: an encoder, a merged decoder,
# and a tokenizer. A segmentation model has none of that — it is one graph plus
# a preprocessor config, and no tokenizer at all. Asking for tokenizer.json
# would abort the run on a file that was never supposed to exist, so the shape
# is chosen per model rather than assumed.
#
# BiRefNet_lite is served fp16, not fp32: both produce the same cut-out (checked
# on a real image before publishing) and fp16 is 109 MB against 213 MB. There is
# no int8 export upstream, so this is the small one.
files_for() {
  case "$1" in
    modnet)
      printf '%s\n' config.json preprocessor_config.json onnx/model_quantized.onnx
      ;;
    # int8 rather than the "quantized" alias: the two are four bytes apart
    # upstream, and int8 is the file that was actually measured.
    pyannote-segmentation-3.0|wespeaker-voxceleb-resnet34-LM)
      printf '%s\n' config.json onnx/model_int8.onnx
      ;;
    *)
      printf '%s\n' "${SUPPORT[@]}" "${WEIGHTS[@]}"
      ;;
  esac
}

# Files that may legitimately be missing upstream, per model. Everything else
# failing is fatal — a silently absent weight file is a model that half-loads.
optional_for() {
  case "$1" in
    modnet) printf '%s\n' ;;
    pyannote-segmentation-3.0|wespeaker-voxceleb-resnet34-LM) printf '%s\n' config.json ;;
    *) printf '%s\n' added_tokens.json ;;
  esac
}

get() { # url dest
  mkdir -p "$(dirname "$2")"
  curl -fSL --retry 3 --retry-delay 2 -o "$2" "$1"
}

echo "==> models"
for repo in "${MODELS[@]}"; do
  m="${repo##*/}"
  optional="$(optional_for "$m")"
  for f in $(files_for "$m"); do
    url="https://huggingface.co/$repo/resolve/main/$f"
    dest="models/$m/$f"
    # added_tokens.json is absent from some exports; every other file is required.
    if ! get "$url" "$dest" 2>/dev/null; then
      rm -f "$dest"
      if printf '%s\n' $optional | grep -qxF "$f"; then
        echo "    $m/$f absent upstream, skipped"
      else
        echo "!!! $m/$f FAILED and is not optional" >&2; exit 1
      fi
      continue
    fi
    printf '    %10d  %s\n' "$(wc -c <"$dest")" "$dest"
  done
done

# ── models that are not on huggingface ───────────────────────────────────────
#
# Everything above is an onnx-community-shaped repo: weights plus a tokenizer,
# fetched by name. These two are single ONNX graphs published elsewhere, so they
# are listed as plain "url -> path" pairs rather than bent into that shape.
#
# Both are for the face and plate blurring tool, and both were picked for their
# LICENCE before their accuracy, because the obvious choice fails on it. Nearly
# every open plate detector is built on Ultralytics YOLOv8/v11, which is
# AGPL-3.0 — publishing those weights next to a static site would put the site
# under AGPL too. These two are MIT, checked against the LICENSE file in each
# repo rather than against a README badge:
#
#   YuNet    opencv/opencv_zoo, MIT, (c) 2020 Shiqi Yu. 232 KB, 75k parameters.
#            Input is a fixed 1x3x640x640 and it emits twelve tensors —
#            cls/obj/bbox/kps at strides 8, 16 and 32 — which the app decodes
#            itself. Measured on a four-person photo: 4/4 faces, 70 ms.
#
#   YOLOv9-t ankandrew/open-image-models, MIT, (c) 2024. 7.8 MB. The "end2end"
#            export, meaning NMS is already inside the graph, so the output is
#            one [n, 7] tensor of finished boxes. The 384px variant, not 640:
#            the extra pixels cost 3x the time and a number plate is a large,
#            high-contrast rectangle that does not need them. Measured on a
#            street photo: the plate, at 0.91, in 105 ms, and no false positive
#            on a photo of four people and no cars.
#
# Only the ONNX graph is served. Neither project's training code is vendored,
# linked or run here.
#   HT-Demucs  166 MB  MIT, StemSplitio/htdemucs-onnx, an ONNX export of Meta's
#              htdemucs. Waveform in, waveform out: one [1,2,343980] tensor of
#              7.8 s stereo becomes [1,4,2,343980], drums/bass/other/vocals. No
#              spectrogram work at all, unlike every masking model.
#
#              It is the first file here big enough to need the splitter above,
#              which is why parts.json has been {} since it was written.
#
#              It also only loads with graph optimisation DISABLED. With the
#              default it dies at session creation with std::bad_alloc, and that
#              wrongly looked like "too big for a browser" for a long time — it
#              is the optimiser's temporaries that blow the wasm heap, not the
#              weights. See split-song/lib/separate.ts.
#
#   GTCRN      535,190 bytes. MIT, (c) 2024 Rong Xiaobin, checked against the
#              LICENSE file at Xiaobin-Rong/gtcrn rather than a badge. 48.2k
#              parameters — the smallest thing this repo serves by three orders
#              of magnitude, and the whole reason /clean-audio can exist.
#
#              Two exports sit side by side upstream: gtcrn.onnx (352,084 B) and
#              gtcrn_simple.onnx (535,190 B), the onnxsim-simplified one. The
#              larger file is the one to serve: upstream's own inference script
#              loads the _simple graph, and both were run here frame-by-frame
#              over 32 frames of tones-plus-noise with identical results — max
#              abs difference 0.0. Same model, so the extra 183 KB buys the
#              graph upstream actually tests against.
#
#              Streaming, not whole-utterance, and that shapes the app code. One
#              STFT frame in: mix [1,257,1,2] at 16 kHz, n_fft 512, hop 256,
#              window hann(512)^0.5. Out: enh, the same shape. Three recurrent
#              caches must be threaded frame to frame — conv_cache
#              [2,1,16,16,33], tra_cache [2,3,1,1,16], inter_cache [2,1,33,16] —
#              all zeros on the first frame. Tensor names and shapes read out of
#              the graph with onnxruntime, not from the paper.
DIRECT=(
  "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx|models/yunet/face_detection_yunet_2023mar.onnx"
  "https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-384-license-plates-end2end.onnx|models/plate/yolo-v9-t-384-license-plates-end2end.onnx"
  "https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/htdemucs_fp16weights.onnx|models/htdemucs/htdemucs_fp16weights.onnx"
  "https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/main/stream/onnx_models/gtcrn_simple.onnx|models/gtcrn/gtcrn_simple.onnx"
)

echo "==> direct downloads"
for entry in "${DIRECT[@]}"; do
  url="${entry%%|*}"
  dest="${entry##*|}"
  get "$url" "$dest"
  # opencv_zoo keeps its weights in LFS, and a pointer file downloads happily
  # with a 200. It is ~130 bytes of ASCII where a model should be, and it fails
  # much later as "invalid protobuf" in someone's browser. Every ONNX file
  # starts with a protobuf field tag, never with "version https://git-lfs".
  if [ "$(wc -c <"$dest")" -lt 10000 ]; then
    echo "!!! $dest is $(wc -c <"$dest") bytes — an LFS pointer or an error page, not a model" >&2
    head -c 120 "$dest" >&2; echo >&2
    exit 1
  fi
  printf '    %10d  %s\n' "$(wc -c <"$dest")" "$dest"
done

echo "==> splitting anything git would refuse"
# Rewritten from nothing every run, so a model dropped from MODELS above cannot
# leave a stale entry behind claiming a file is split when it is gone.
python3 - "$PART_MIB" <<'PY'
import json, pathlib, sys

part_bytes = int(sys.argv[1]) * 1024 * 1024
models = pathlib.Path("models")

# Yesterday's pieces, before anything is measured. A part left over from a run
# with a different PART_MIB would otherwise survive next to the new ones and be
# served as though it belonged.
for stale in models.rglob("*.onnx.part*"):
    stale.unlink()

manifest = {}
for weights in sorted(models.rglob("*.onnx")):
    total = weights.stat().st_size
    if total < part_bytes:
        continue

    count = 0
    with weights.open("rb") as src:
        while chunk := src.read(part_bytes):
            (weights.parent / f"{weights.name}.part{count}").write_bytes(chunk)
            count += 1

    # Read back what was written rather than trusting the arithmetic: this is
    # the one place where being wrong produces a model that loads and is subtly
    # corrupt, which is far worse than one that fails outright.
    rejoined = b"".join(
        (weights.parent / f"{weights.name}.part{i}").read_bytes() for i in range(count)
    )
    if rejoined != weights.read_bytes():
        sys.exit(f"!!! {weights} does not survive a split/rejoin round trip")

    weights.unlink()
    key = str(weights.relative_to(models))
    manifest[key] = {"parts": count, "bytes": total}
    print(f"    {total:10d}  {key} -> {count} parts")

(models / "parts.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"    {len(manifest)} split file(s) recorded in models/parts.json")
PY

echo "==> onnxruntime-web $ONNX_VERSION"
# Both variants, because transformers.js picks between them at runtime: the
# plain build for Safari, asyncify everywhere else. Serving only one would
# silently break whichever half of the world got the other.
mkdir -p "ort/$ONNX_VERSION"
# The jsep pair is new and is not optional for anything that wants WebGPU.
# Requesting device:"webgpu" against the plain build fails at session creation
# with "no available backend found. ERR: [webgpu] TypeError:
# eP(...).webgpuInit is not a function" — the WebGPU execution provider lives in
# the JSEP build and nowhere else. JSEP also still runs on plain wasm, so it is
# a superset rather than an alternative.
for f in \
  ort-wasm-simd-threaded.mjs \
  ort-wasm-simd-threaded.wasm \
  ort-wasm-simd-threaded.asyncify.mjs \
  ort-wasm-simd-threaded.asyncify.wasm \
  ort-wasm-simd-threaded.jsep.mjs \
  ort-wasm-simd-threaded.jsep.wasm
do
  cp "$ORT_SRC/$f" "ort/$ONNX_VERSION/$f"
  printf '    %10d  ort/%s/%s\n' "$(wc -c <"ort/$ONNX_VERSION/$f")" "$ONNX_VERSION" "$f"
done


# ── duckdb-wasm ──────────────────────────────────────────────────────────────
#
# 34 MB, which is why it is here and not in the tools repo's public/ — that
# would commit 34 MB into the app repo and ship it on every build.
#
# The "eh" bundle, not "coi": coi is the cross-origin-isolated build and needs
# SharedArrayBuffer, which needs COOP/COEP headers GitHub Pages cannot send.
# eh is the single-threaded exception-handling build and is what this project
# can actually run. mvp is the older fallback and is 5 MB larger for less.
DUCKDB_SRC="../tools/node_modules/@duckdb/duckdb-wasm/dist"
if [ -d "$DUCKDB_SRC" ]; then
  DUCKDB_VERSION="$(python3 -c "import json;print(json.load(open('../tools/node_modules/@duckdb/duckdb-wasm/package.json'))['version'])")"
  echo "==> duckdb-wasm $DUCKDB_VERSION"
  mkdir -p "duckdb/$DUCKDB_VERSION"
  for f in duckdb-eh.wasm duckdb-browser-eh.worker.js; do
    cp "$DUCKDB_SRC/$f" "duckdb/$DUCKDB_VERSION/$f"
    printf '    %10d  duckdb/%s/%s\n' "$(wc -c <"duckdb/$DUCKDB_VERSION/$f")" "$DUCKDB_VERSION" "$f"
  done
else
  echo "==> duckdb-wasm skipped (not installed in ../tools)"
fi

echo "==> total $(du -sh models ort duckdb 2>/dev/null | awk '{print $1" "$2}' | tr '\n' ' ')"
