#!/bin/sh
# Deterministic cache key over the inputs that force a Rust rebuild.
#
# `hashFiles()` cannot be used for this: GitHub evaluates it when the step
# runs, and oxide_build.py writes <crate>.ptx into kernels-oxide/ before the
# save step — so the save key hashed build artifacts while the restore key
# (a clean checkout) did not, and the two never matched. `git ls-tree` reads
# the committed blobs instead: identical in both workflows, immune to
# anything a build step drops in the working tree.
#
# Prints one hex digest. Both native.yml (save) and kernels.yml (restore) call
# this, so the key cannot drift between them.
set -eu
git ls-tree -r HEAD -- src kernels-oxide Cargo.toml Cargo.lock pyproject.toml \
  | sha256sum | cut -d' ' -f1
