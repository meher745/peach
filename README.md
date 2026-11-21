# Shopping Sessions: Rust pipeline

Single static-ish Rust binary (`src/main.rs`), no engine. `rules.md` is the spec.

Pipeline:

1. Links: parallel parse, union-find, every linked ID mapped to its person's byte-wise smallest ID.
2. Events and clicks: files read in 8 MB chunks with `pread`, lines split with a SIMD bitmask
   scanner (comma / newline / quote masks), timestamps parsed with SWAR checks. Events are
   validated and hash-partitioned by `event_id`; clicks by person.
3. Dedup per `event_id` partition (smallest `ingest_ts` copy wins, then its validity decides),
   survivors re-partitioned by person.
4. Per person partition: sort events by time, sessionize (gap > 30 min), credit each session to
   the latest click in the 7 days up to its start (ties: smallest `click_id`), count assists with
   a sliding window of active campaigns, aggregate per person/day and campaign/day.
5. Write Parquet with arrow-rs.

IDs become `u128` keys: strings of at most 15 bytes are packed exactly (key order equals
byte-wise order); longer ones are hashed and their bytes kept in a side table.

Local checks:

```sh
cargo build --release
python3 scripts/run_fixtures.py fixtures -- target/release/solution
python3 scripts/reference.py DATA_DIR OUT_DIR   # slow literal Python reference, for cross-checks
make build && make check-dev
```
