export DYLD_LIBRARY_PATH=./prebuilt/macos_arm64

./bazel-bin/runtime/engine/litert_lm_advanced_main \
      --model_path=$HOME/models/Gemma3-1B-IT/gemma3-1b-it-int4.litertlm \
      --backend=gpu \
      --input_prompt="Write a short story about a robot" \
      --benchmark=true \
      --max_num_tokens=512 2>&1 | grep -E "BenchmarkInfo|Prefill|Decode|tokens/sec|Init|Time to first|Total init|^--"
