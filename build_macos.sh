#!/bin/bash
#
# Build script for LiteRT-LM on macOS
#
# This script builds the following targets:
#   - //c:engine                              : C API library (libengine.a)
#   - //runtime/engine:litert_lm_main         : CLI executable for LLM inference
#   - //runtime/engine:litert_lm_advanced_main: CLI executable with advanced features
#
# Build options:
#   --define=litert_link_capi_so=true         : Required for GPU support
#   --define=resolve_symbols_in_exec=false    : Required for GPU support (overrides .bazelrc default)
#
# Output files will be located in:
#   - bazel-bin/c/libengine.a
#   - bazel-bin/runtime/engine/litert_lm_main
#   - bazel-bin/runtime/engine/litert_lm_advanced_main
#

bazelisk build //c:engine //runtime/engine:litert_lm_main //runtime/engine:litert_lm_advanced_main \
    --define=litert_link_capi_so=true \
    --define=resolve_symbols_in_exec=false
