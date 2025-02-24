import math
from typing import BinaryIO  # noqa: F401

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from modal import App, Image, asgi_app, method, enter

stub = App("esm")
web_app = FastAPI()

image = (
    Image.debian_slim(python_version="3.11")
    .micromamba()
    .apt_install("wget", "git")
    .pip_install(
        "torch",
        "fair-esm",
    )
    .run_commands(
        "wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
    )
)

with image.imports():
    import esm
    import torch


@stub.cls(image=image, gpu="a100")
class ESMModel:
    @enter()
    async def load_model(self):
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval().cuda()

        return self.model

    @method()
    async def compute_perplexity(self, json_data: dict):
        data = [("sequence", json_data["protein_sequence"])]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        log_probs = []
        for i in range(1, len(json_data["protein_sequence"]) - 1):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, i] = self.alphabet.mask_idx
            with torch.no_grad():
                token_probs = torch.log_softmax(
                    self.model(batch_tokens_masked.cuda())["logits"], dim=-1
                )
            log_probs.append(
                token_probs[
                    0, i, self.alphabet.get_idx(json_data["protein_sequence"][i])
                ].item()
            )
        return {"perplexity": sum(log_probs)}

    @method()
    async def get_probabilities(self, json_data: dict) -> dict:
        mutate_sequence = {}
        position = json_data["position"]
        data = [("sequence", json_data["protein_sequence"])]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, position] = self.alphabet.mask_idx
        with torch.no_grad():
            logits = torch.log_softmax(
                self.model(batch_tokens_masked.cuda())["logits"], dim=-1
            )
        probabilities = torch.nn.functional.softmax(logits[0, position], dim=0)
        log_probabilities = torch.log(probabilities)
        wt_residue = batch_tokens[0, position].item()
        log_prob_wt = log_probabilities[wt_residue].item()
        for amino_acid in amino_acids:
            log_prob_mt = log_probabilities[self.alphabet.get_idx(amino_acid)].item()
            mutate_sequence[amino_acid] = log_prob_mt - log_prob_wt
        mutate_seq_des = (
            "Log Likelihood Ratios(LLR) at this site for each amino acid type are: "
        )
        for key, value in mutate_sequence.items():
            mutate_seq_des += f"{key}: {value}, "
        mutate_seq_des += (
            f"with the wild type residue being {self.alphabet.get_tok(wt_residue)}, higher Log Likelihood Ratios could "
            f"indicate beneficial mutations"
        )
        return mutate_sequence

    @method()
    async def compute_pll_correct(self, sequence: str) -> dict:
        data = [("protein", sequence)]
        batch_converter = self.alphabet.get_batch_converter()
        *_, batch_tokens = batch_converter(data)
        log_probs = []
        for i in range(len(sequence)):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, i + 1] = self.alphabet.mask_idx
            with torch.no_grad():
                token_probs = torch.log_softmax(
                    self.model(batch_tokens_masked.to("cuda:0"))["logits"], dim=-1
                )
            log_probs.append(
                token_probs[0, i + 1, self.alphabet.get_idx(sequence[i])].item()
            )
        return {"pll": math.fsum(log_probs)}


@stub.function(image=image, gpu="a10g")
def predict(json_data: dict) -> dict:
    mut_sequence = json_data["mut_sequence"]
    wt_sequence = json_data["wt_sequence"]
    mut_perplexity = ESMModel().compute_perplexity.remote(mut_sequence)
    wt_perplexity = ESMModel().compute_perplexity.remote(wt_sequence)
    if mut_perplexity < wt_perplexity:
        return {
            "message": "The proposed mutated sequence has lower pseudo perplexity than the wild type sequence."
        }
    return {
        "message": "The proposed mutated sequence has higher pseudo perplexity than the wild type sequence."
    }


@stub.function(image=image, gpu="a10g")
async def run_compute_perplexity(sequence: str, seq_id: str) -> dict:  # noqa: RUF029
    perplexity = ESMModel().compute_perplexity.remote({"protein_sequence": sequence})
    return {seq_id: perplexity}


@web_app.post("/sample/esm")
async def endpoint(
    json_data: dict, name:str
):
    print(json_data, name)
    if name == "perplexity":
        blob = await predict.remote.aio(json_data)
    elif name == "probabilities":
        blob = ESMModel().get_probabilities.remote(json_data)
    elif name == "pll":
        blob = ESMModel().compute_pll_correct.remote(json_data["sequence"])
    else:
        blob = {"message": "No name found"}
    return JSONResponse(content=blob)


@web_app.get("/")
async def root() -> dict:
    return {"message": "ESM Model"}


@stub.function()
@asgi_app()
def app() -> FastAPI:
    return web_app