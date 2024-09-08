@REM NOTES:
@REM cargo install simple-http-server
@REM cargo update -p wasm-bindgen
@REM cargo install wasm-pack --no-default-features
@REM wasm-opt "pkg/guardian_ai_bg.wasm" -O2 --fast-math -o "pkg/guardian_ai_bg_opt.wasm"
@REM set RUSTFLAGS=-C target-feature=+atomics,+bulk-memory,+mutable-globals --cfg=web_sys_unstable_apis

@REM cargo build --lib --release --target wasm32-unknown-unknown --package guardian_ai -Z build-std=std,panic_abort
@REM wasm-bindgen --target web --no-typescript --out-dir "guardian_website/pkg" %CARGO_TARGET_DIR%/wasm32-unknown-unknown/release/guardian_ai.wasm
@REM python fix_dispatch.py
@REM wasm-pack build --target web --no-typescript "guardian_ai" --out-dir "../guardian_website/pkg" --dev -- -Z build-std=panic_abort,std

@REM: RUN
wasm-pack build --target web --no-typescript "guardian_ai" --out-dir "../guardian_website/pkg" --dev -- -Z build-std=panic_abort,std
python fix_wgpu.py
cargo run --package guardian_website --target "x86_64-pc-windows-msvc"
