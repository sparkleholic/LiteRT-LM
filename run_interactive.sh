export DYLD_LIBRARY_PATH=./prebuilt/macos_arm64

./bazel-bin/runtime/engine/litert_lm_advanced_main \
      --model_path=$HOME/models/Gemma3-1B-IT/gemma3-1b-it-int4.litertlm \
      --backend=gpu \
      --log_sink_file=/tmp/litert_lm.log \
      --multi_turns=true 2>/dev/null
