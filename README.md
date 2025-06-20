## Huggingface Workspace

This is a simple setup for running ad-hoc experiments with a large selection of models loaded through `transformers` from `Huggingface`.
The base image is built on top of `nvidia/cuda:12.4.0-devel-ubuntu22.04`:
- CUDA `V12.4.131`
- Torch `2.6.0`
- Transformers `4.52.4`
- Flash Attention `2.7.3`

## Docker Compose

The container is based on a simple image that has `transformers` `accelerate` and `flash-attention2` already built and ready.

Use `docker compose up -d` to start the container. You can access the container by using `docker exec -it huggingface-workspace bash` or inspecting and using the Exec tab the container using Docker Desktop.

To test the whole system do:
```
docker compose up -d
docker exec -it huggingface-workspace bash
python3 scripts/test.py
```

## Workflow

The following folders are treated as shared volumes with the container, allowing for writing scripts, loading data, and saving results seamlessly between the container and the desktop.

You can find the following folders:
- data: For loading data.
- models: Reserved for loading models locally. HF models are saved here so you don't have to redownload them every time.
- output: For saving data.
- scripts: Scripts to execute. You can find some scaffolding and an example for a small multimodal LLM here.

To start a script in the background (no interactive terminal do) `docker exec -d huggingface-workspace python3 scripts/<script_name>`

## llm_base.py and New Models

`VisionLanguageModelPrototype` is a simple wrapper that mainly allows you to work with LLMs in a `with ...` syntax and automatically unload them after you finish. Useful for running experiments on multiple LLMs.

`qwen.py` shows a simple example for loading a model and setting up inference. Check the model card on HuggingFace, it can be slightly different for different models.

This setup is for running batches of experiments with `model.generate`, it does not serve the model for streaming and ad-hoc queries.
