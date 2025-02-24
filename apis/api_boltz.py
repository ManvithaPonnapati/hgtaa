from dataclasses import dataclass
from pathlib import Path

import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

# A constant for timeout calculations
MINUTES = 60

# Create a Modal App and a FastAPI web app
stub = modal.App(name="boltz1-api")
web_app = FastAPI()

# Define a Modal image that installs boltz and PyYAML
image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        "uv pip install --system --compile-bytecode boltz==0.3.2",
        "pip install pyyaml"
    )
)

# Create a Modal Volume for model weights and assign a directory path
boltz_model_volume = modal.Volume.from_name("boltz1-models", create_if_missing=True)
models_dir = Path("/models/boltz1")

# Define a separate image for downloading model weights via huggingface_hub
download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]==0.26.3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


# Define a dataclass for MSA information.
@dataclass
class MSA:
    data: str
    path: Path


# Helper function to package outputs into a tar.gz archive.
def package_outputs(output_dir: str) -> bytes:
    import io
    import tarfile

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        tar.add(output_dir, arcname=output_dir)
    return tar_buffer.getvalue()


# Modal function to download model weights from Hugging Face.
@stub.function(
    volumes={models_dir: boltz_model_volume},
    timeout=20 * MINUTES,
    image=download_image,
)
def download_model(force_download: bool = False, revision: str = "7c1d83b779e4c65ecc37dfdf0c6b2788076f31e1"):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="boltz-community/boltz-1",
        revision=revision,
        local_dir=str(models_dir),
        force_download=force_download,
    )
    boltz_model_volume.commit()
    print(f"🧬 Model downloaded to {models_dir}")


# Modal function that runs the boltz1 inference.
@stub.function(
    image=image,
    volumes={models_dir: boltz_model_volume},
    timeout=10 * MINUTES,
    gpu="H100",
)
def boltz1_inference(boltz_input_yaml: str, msas: list[MSA], args: str = "") -> bytes:
    import shlex
    import subprocess

    # Write the provided YAML input to a temporary file.
    input_path = Path("input.yaml")
    input_path.write_text(boltz_input_yaml)

    # For every provided MSA, write its data to the corresponding file.
    for msa in msas:
        msa.path.write_text(msa.data)

    # Process additional command-line arguments.
    args_list = shlex.split(args)

    print(f"🧬 Running boltz with input file {input_path} using model weights at {models_dir}")
    subprocess.run(
        ["boltz", "predict", str(input_path), "--cache", str(models_dir)] + args_list,
        check=True,
    )

    print("🧬 Packaging up outputs")
    # Assume that the output directory is named based on the input file name (without suffix).
    output_dir = f"boltz_results_{input_path.with_suffix('').name}"
    output_bytes = package_outputs(output_dir)
    return output_bytes


# FastAPI endpoint for inference.
@web_app.post("/infer")
async def infer_endpoint(payload: dict):
    """
    Expected JSON payload structure:
    {
      "input_yaml": "<YAML content as a string>",
      "args": "<optional boltz predict command args>",
      "force_download": <optional boolean, default false>,
      "msas": [
          {"msa_path": ".path/to/msa.a3m", "data": "<MSA file contents>"},
          ...
      ]
    }
    """
    input_yaml = payload.get("input_yaml")
    if not input_yaml:
        raise HTTPException(status_code=400, detail="Missing 'input_yaml' in request payload.")
    args = payload.get("args", "")
    force_download = payload.get("force_download", False)
    msas_payload = payload.get("msas", [])
    msas = []
    for item in msas_payload:
        if "msa_path" not in item or "data" not in item:
            raise HTTPException(status_code=400, detail="Each MSA must include 'msa_path' and 'data'.")
        msas.append(MSA(data=item["data"], path=Path(item["msa_path"])))

    # Optionally refresh the model weights if requested.
    if force_download:
        await download_model.remote.aio(force_download=True)

    # Run the boltz1 inference Modal function.
    output_bytes = await boltz1_inference.remote.aio(
        boltz_input_yaml=input_yaml,
        msas=msas,
        args=args
    )
    return Response(content=output_bytes, media_type="application/gzip")


# A simple root endpoint.
@web_app.get("/")
async def root():
    return {"message": "Boltz1 Inference API"}


# Expose the FastAPI app via Modal's ASGI integration.
@stub.function()
@modal.asgi_app()
def app() -> FastAPI:
    return web_app
