FROM rust:1.100-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main(){}' > src/main.rs && cargo fetch --locked
COPY src ./src
# Scorers are Cascade Lake or newer: AVX2/BMI2/POPCNT are safe. With --target,
# RUSTFLAGS apply only to our binary, not to build scripts.
ARG TARGET_CPU=x86-64-v3
RUN touch src/main.rs && RUSTFLAGS="-C target-cpu=${TARGET_CPU}" cargo build --release --locked --target x86_64-unknown-linux-gnu \
    && cp target/x86_64-unknown-linux-gnu/release/solution /solution && strip /solution

FROM debian:bookworm-slim
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app
COPY --from=build /solution /app/solution
# keep freed memory instead of returning it to the OS (avoids re-zeroing pages)
ENV MIMALLOC_PURGE_DELAY=-1 MIMALLOC_ALLOW_LARGE_OS_PAGES=1 THREADS=12
USER 1000
CMD ["/app/solution"]
