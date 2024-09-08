@REM NOTES:
@REM --node --firefox --chrome --safari --headless
@REM cargo test --target wasm32-unknown-unknown  # Does not work

@REM .cargo/Config.toml needs to be empty to run these tests :((

@REM RUN: 
cargo test

@REM HEADLESS:
wasm-pack test --firefox --chrome --headless guardian_io
wasm-pack test --firefox --chrome --headless guardian_process
