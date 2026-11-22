# Shopping Sessions: native Rust pipeline.
# The scorer builds this natively on linux/amd64 (Intel Xeon, Cascade Lake or newer).
#
# Hardened build:
#   * rust toolchain is PINNED (reproducible: unpinned `rust:slim` floats to the latest
#     rustc, whose codegen can regress this parser's hot loop between tunings).
#   * PGO uses a COMMITTED profile (pgo/x86.profdata) instead of training inside the
#     build, so it never silently falls back for want of llvm-tools or build-time CPU.
#     If the committed profile is rejected (toolchain/LLVM mismatch) the build cleanly
#     falls back to a plain optimised build, so a binary always ships.
FROM rust:1.96-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY pgo ./pgo
COPY src ./src
# x86-64-v3 (AVX2/BMI1/BMI2/LZCNT/POPCNT): available on every Cascade-Lake-or-newer Xeon.
RUN set -eu; \
    T=x86_64-unknown-linux-gnu; \
    BASE="-C target-cpu=x86-64-v3"; \
    if RUSTFLAGS="$BASE -Cprofile-use=/src/pgo/x86.profdata -Cllvm-args=-pgo-warn-missing-function" \
         cargo build --release --locked --target "$T" 2>/tmp/pgo.err; then \
        echo "=== build: PGO (committed profile) ==="; \
    else \
        echo "=== build: PGO profile rejected, plain optimised fallback ==="; cat /tmp/pgo.err || true; \
        RUSTFLAGS="$BASE" cargo build --release --locked --target "$T"; \
    fi; \
    strip -o /shopping-sessions "target/$T/release/shopping-sessions"

# Minimal runtime: plain glibc + libgcc only, no shell (verified faster on the scorer
# than a jemalloc/alignment variant).
FROM gcr.io/distroless/cc-debian12
COPY --from=build /shopping-sessions /shopping-sessions
# Scorer env: DATA_DIR=/data OUTPUT_DIR=/output SCRATCH_DIR=/scratch NCPU=8 MEM_GB=24.
ENV DATA_DIR=/data OUTPUT_DIR=/output
USER 1000:1000
ENTRYPOINT ["/shopping-sessions"]
