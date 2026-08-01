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
# Why these two model sizes and no third: whisper-small's merged decoder is
# 156 MB and GitHub hard-blocks any file over 100 MiB on push. Git LFS is the
# documented answer and it is not an option here, because GitHub Pages does not
# serve LFS content. So 100 MiB per file is a real ceiling, not a preference.
#
# Why multilingual and not the .en variants: identical byte size. whisper-base
# and whisper-base.en both quantise to 23,201,3xx + 53,69x,xxx. The multilingual
# model handles 99 languages for the same download, so the .en builds have
# nothing to offer.

set -euo pipefail
cd "$(dirname "$0")"

ONNX_VERSION="1.26.0-dev.20260416-b7804b056c"
ORT_SRC="../tools/node_modules/onnxruntime-web/dist"

MODELS=(whisper-tiny whisper-base)

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

get() { # url dest
  mkdir -p "$(dirname "$2")"
  curl -fSL --retry 3 --retry-delay 2 -o "$2" "$1"
}

echo "==> models"
for m in "${MODELS[@]}"; do
  for f in "${SUPPORT[@]}" "${WEIGHTS[@]}"; do
    url="https://huggingface.co/onnx-community/$m/resolve/main/$f"
    dest="models/$m/$f"
    # added_tokens.json is absent from some exports; every other file is required.
    if ! get "$url" "$dest" 2>/dev/null; then
      rm -f "$dest"
      case "$f" in
        added_tokens.json) echo "    $m/$f absent upstream, skipped" ;;
        *) echo "!!! $m/$f FAILED and is not optional" >&2; exit 1 ;;
      esac
      continue
    fi
    printf '    %10d  %s\n' "$(wc -c <"$dest")" "$dest"
  done
done

echo "==> onnxruntime-web $ONNX_VERSION"
# Both variants, because transformers.js picks between them at runtime: the
# plain build for Safari, asyncify everywhere else. Serving only one would
# silently break whichever half of the world got the other.
mkdir -p "ort/$ONNX_VERSION"
for f in \
  ort-wasm-simd-threaded.mjs \
  ort-wasm-simd-threaded.wasm \
  ort-wasm-simd-threaded.asyncify.mjs \
  ort-wasm-simd-threaded.asyncify.wasm
do
  cp "$ORT_SRC/$f" "ort/$ONNX_VERSION/$f"
  printf '    %10d  ort/%s/%s\n' "$(wc -c <"ort/$ONNX_VERSION/$f")" "$ONNX_VERSION" "$f"
done

echo "==> total $(du -sh models ort | awk '{print $1" "$2}' | tr '\n' ' ')"
