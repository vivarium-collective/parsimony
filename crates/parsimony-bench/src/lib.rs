//! Cross-language benchmarking harness comparing parsimony against
//! cellPACK Python (clone at `/home/pattern/code/cellpack`).
//!
//! See `docs/parsimony-design.md` §13 for the validation strategy.
//!
//! Phase 2/3 module plan (not yet implemented):
//!
//! ```ignore
//! pub mod harness;       // run both engines on a recipe, collect timings
//! pub mod metrics;       // pair-correlation, NN distance, void-size distributions
//! pub mod cellpack_py;   // subprocess wrapper around `python -m cellpack.bin.pack`
//! pub mod report;        // CSV / human-readable summary
//! ```
