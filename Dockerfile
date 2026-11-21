FROM rust:1.100-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main(){}' > src/main.rs && cargo fetch --locked
COPY src ./src
RUN touch src/main.rs && cargo build --release --locked && strip target/release/solution

FROM debian:bookworm-slim
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app
COPY --from=build /src/target/release/solution /app/solution
USER 1000
CMD ["/app/solution"]
