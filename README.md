# oss-llm-tools
OSS LLM Tools for Conversion Eval Numerical Degging and Benchmarking

## lm-eval bisect tools

Find first bad commit that dropped the accuracy of a model.

```
cd <target repo>
python ../oss-llm-tools/bisect_accuracy.py --good <Good Commit> --bad <Bad Commit> --model google/gemma-3-12b-it --task gsm8k --target 0.4 --limit 100 --bisect_log_file --model_args '{"tensor_parallel_size": 4}'  --eval_args '{"num_fewshot":5}' --stop_with_exception --bisect_log /tmp/bisect.log

```
