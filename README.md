# Shopping Sessions: Rust pipeline

Single Rust binary (`src/main.rs`), no engine. `rules.md` is the spec.

Pipeline:

1. Parse (all threads, one shared task list: link files first, then events, then clicks).
   Files are mmapped and cut into 8 MB chunks; lines are split with a SIMD bitmask scanner
   working in 32 KB windows, quotes checked branch-free. Timestamps: window dates come from
   a small `YYYY-MM-DD` table, times and F3 digits are parsed with SWAR.
   - Events: each valid event is appended to its worker's record array (24 bytes, in parse
     order); every event with an `event_id` also writes a 16-byte tuple
     `(event_id, ingest, location)` into one of 256 `event_id` partitions.
   - Links: a dedicated thread runs union-find once every link chunk is parsed, while events
     are still being parsed. Clicks wait for it and are partitioned by person.
2. Dedup per `event_id` partition, split into cache-sized pieces: the first copy is held in a
   flat table, every later copy either loses or replaces it (smaller `ingest_ts`), and the
   loser's alive byte is cleared. Invalid copies take part, so an invalid earliest copy drops
   the event.
3. Person shuffle: each worker resolves the person of its alive records (bit filter, then a
   prefetched open-addressing link table), counts per person partition, and scatters into
   one shared array, so partitions are contiguous.
4. Per person partition: radix sort (3 × 11-bit ts digits, then dense person id), sessions
   (gap > 30 min), credit to the latest click in the 7 days up to the start (ties: smallest
   `click_id`), assists from a sliding window of active campaigns, per person/day and
   campaign/day aggregates.
5. Parquet via arrow-rs.

IDs become exact u64 keys: strings of at most 10 bytes from `-0-9A-Z_a-z` are packed 6 bits
per character (key order equals byte order); others get a 60-bit hash and their bytes are
kept in a side table.

Local checks:

```sh
cargo build --release
python3 scripts/run_fixtures.py fixtures -- target/release/solution
python3 scripts/reference.py DATA_DIR OUT_DIR   # slow literal Python reference, for cross-checks
make build && make check-dev
```
