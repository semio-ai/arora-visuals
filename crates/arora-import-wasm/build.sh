#!/usr/bin/env bash
# Builds the wasm module into the npm package, at `packages/import/pkg/`,
# which the package's wrapper imports.
#
# The workspace's release profile (opt "s", fat LTO, one codegen unit), since
# the module is downloaded, then binaryen's `wasm-opt -Oz` when WASM_OPT names
# it (or it is on PATH).
#
# Needs wasm-bindgen-cli at exactly the version Cargo.lock pins for the
# wasm-bindgen crate; set WASM_BINDGEN to its path if it is not on PATH.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
bindgen="${WASM_BINDGEN:-wasm-bindgen}"
wasm_opt="${WASM_OPT:-$(command -v wasm-opt || true)}"
cargo build --locked --release --target wasm32-unknown-unknown -p arora-import-wasm \
  --manifest-path "$root/Cargo.toml"
out="$root/packages/import/pkg"
rm -rf "$out"
"$bindgen" --target web --remove-name-section --remove-producers-section \
  --out-dir "$out" "$root/target/wasm32-unknown-unknown/release/arora_import_wasm.wasm"
if [ -n "$wasm_opt" ]; then
  # The features rustc's wasm32 target emits by default; wasm-opt must be
  # told, or it refuses the module.
  "$wasm_opt" -Oz --enable-bulk-memory --enable-mutable-globals \
    --enable-nontrapping-float-to-int --enable-sign-ext \
    --enable-reference-types --enable-multivalue \
    "$out/arora_import_wasm_bg.wasm" -o "$out/arora_import_wasm_bg.wasm"
fi
ls -l "$out"
