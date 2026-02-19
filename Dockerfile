# Stage 1: Build
FROM rust:1-slim-bookworm AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY benches/ benches/
COPY tests/ tests/

RUN cargo build --release --bin gridline-forge

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/gridline-forge /usr/local/bin/gridline-forge
