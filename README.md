# NESTFUL
This is the official repository for `NESTFUL`.
- Paper Title: **_NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls_**
- Link: https://arxiv.org/abs/2409.03797v2
- HuggingFace Data Link: https://huggingface.co/datasets/ibm-research/nestful

### Data
We have shared the latest NESTFUL evaluation set under `data_v2` dir.
- `nestful_data.jsonl`: It has 1861 evaluation data for nested sequencing.
- `executable_functions`: Contains the implementation of all the functions in the benchmark.

The `data_v1` directory includes the data for the previous version of the paper - [link](https://arxiv.org/abs/2409.03797v1).
- `executable`: contains data and spec with necessary information to execute them through RapidAPI.
- `non-executable`: contains the nested sequencing data from SGD and GLAIVE that are hand-picked by human annotators from data synthetically generated using an LLM.
