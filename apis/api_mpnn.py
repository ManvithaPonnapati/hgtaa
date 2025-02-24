import tempfile

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from modal import App, Image, asgi_app, gpu

# use mid-tier GPU here
gpu_config = gpu.A10G(count=1)

app = App("colabdesign")
web_app = FastAPI()

image = (
    Image.debian_slim(python_version="3.11")
    .micromamba()
    .apt_install("wget", "git")
    .pip_install("git+https://github.com/sokrypton/ColabDesign.git@v1.1.1")
    .pip_install("google-cloud-storage")
    .run_commands(
        'pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
        gpu=gpu_config,
    )  # this is the version used by the model
    .pip_install("google-cloud-storage", "google-auth")
)
with image.imports():
    from colabdesign.mpnn import mk_mpnn_model


@app.function(
    image=image,
    gpu=gpu_config,
    concurrency_limit=5,
)
def sample(
    pdb_string: str, mpnn_config: dict
) -> list[dict]:
    model = mk_mpnn_model()
    fix_pos = str(mpnn_config["fix_pos"])
    inverse = bool(mpnn_config["inverse"])
    sampling_temp = float(mpnn_config["temp"])
    batch = int(mpnn_config["batch"])  # Samples sequences in parallel
    chains_to_design = mpnn_config["chains"]
    with tempfile.NamedTemporaryFile(
        delete=True, mode="w+", suffix=".pdb"
    ) as temp_file:
        temp_file.write(pdb_string)
        temp_file.flush()
        model.prep_inputs(
            pdb_filename=temp_file.name,
            chain=chains_to_design,
            inverse=inverse,
            fix_pos = fix_pos
        )
    out = model.sample_parallel(temperature=sampling_temp, batch=batch)
    data = [
        {
            "score": out["score"][n],
            "seqid": out["seqid"][n],
            "seq": out["seq"][n],
        }
        for n in range(batch)
    ]
    return data


@app.function(image=image)
def generate_sequences(pdb_string: str, mpnn_config: dict):
    arguments_list = [
        (pdb_string, mpnn_config)
    ]
    return sample.starmap(arguments_list)


@web_app.post("/sample")
async def app_endpoint(
    json_data: dict
):
    mpnn_config = json_data["params"]
    pdb_string = json_data["pdb_string"]
    return JSONResponse(
        content=await generate_sequences.remote.aio(pdb_string, mpnn_config)
    )


@web_app.get("/")
async def root() -> dict:
    return {"message": "ProteinMPNN ColabDesign"}


@app.function()
@asgi_app()
def fastapi_app():
    return web_app
