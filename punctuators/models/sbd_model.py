import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Union, Tuple

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from sentencepiece import SentencePieceProcessor
from torch.utils.data import DataLoader

from punctuators.collectors import SBDResultCollector
from punctuators.data import TextInferenceDataset


@dataclass
class SBDConfig:
    """ No abstraction yet, will be refactored """

    pass


@dataclass
class SBDConfigONNX(SBDConfig):
    spe_filename: str
    model_filename: str
    config_filename: str = "config.yaml"
    hf_repo_id: Optional[str] = None
    directory: Optional[str] = None


class SBDModel:
    """ No abstraction yet, will be refactored """

    def __init__(self):
        pass


class SBDModelONNX(SBDModel):
    def __init__(self, cfg: SBDConfigONNX):
        super().__init__()
        if cfg.hf_repo_id is not None:
            self._spe_path = hf_hub_download(repo_id=cfg.hf_repo_id, filename=cfg.spe_filename)
            onnx_path = hf_hub_download(repo_id=cfg.hf_repo_id, filename=cfg.model_filename)
            config_path = hf_hub_download(repo_id=cfg.hf_repo_id, filename=cfg.config_filename)
        else:
            if cfg.directory is None:
                raise ValueError(f"Need HF repo ID or local directory name; got {cfg}")
            self._spe_path = os.path.join(cfg.directory, cfg.spe_filename)
            onnx_path = os.path.join(cfg.directory, cfg.model_filename)
            config_path = os.path.join(cfg.directory, cfg.config_filename)
        self._tokenizer: SentencePieceProcessor = SentencePieceProcessor(self._spe_path)  # noqa
        self._ort_session: ort.InferenceSession = ort.InferenceSession(onnx_path)
        self._config = OmegaConf.load(config_path)
        self._max_len = self._config.max_length
        self._languages: List[str] = self._config.languages

    @classmethod
    def pretrained_model_info(cls) -> Dict[str, SBDConfigONNX]:
        info = {
            "sbd_multi_lang": SBDConfigONNX(
                hf_repo_id="1-800-BAD-CODE/sentence_boundary_detection_multilang",
                spe_filename="spe_mixed_case_64k_49lang.model",
                model_filename="sbd_49lang_bert_small.onnx",
            ),
        }
        return info

    @classmethod
    def from_pretrained(cls, pretrained_name: str) -> "SBDModelONNX":
        available_models: Dict[str, SBDConfigONNX] = cls.pretrained_model_info()
        if pretrained_name not in available_models:
            raise ValueError(
                f"Pretrained name '{pretrained_name}' not in available models: '{list(available_models.keys())}'"
            )
        return cls(cfg=available_models[pretrained_name])

    @property
    def languages(self) -> List[str]:
        return self._languages

    def _setup_dataloader(
        self, texts: List[str], batch_size_tokens: int, overlap: int, num_workers: int = os.cpu_count() - 1
    ) -> DataLoader:
        dataset: TextInferenceDataset = TextInferenceDataset(
            texts=texts,
            batch_size_tokens=batch_size_tokens,
            overlap=overlap,
            max_length=self._max_len,
            spe_model_path=self._spe_path,
        )
        return DataLoader(
            dataset=dataset, num_workers=num_workers, collate_fn=dataset.collate_fn, batch_sampler=dataset.sampler,
        )

    @torch.inference_mode()
    def infer(
        self,
        texts: List[str],
        threshold: float = 0.5,
        batch_size_tokens: int = 4096,
        overlap: int = 16,
        num_workers: int = 0,
    ) -> Union[List[str], List[List[str]]]:
        # Collect results for each input (long inputs will be split by the data loader)
        collectors: List[SBDResultCollector] = [
            SBDResultCollector(sp_model=self._tokenizer, overlap=overlap) for _ in range(len(texts))
        ]
        # Instantiate a data loader with the texts
        dataloader: DataLoader = self._setup_dataloader(
            texts=texts, batch_size_tokens=batch_size_tokens, overlap=overlap, num_workers=num_workers
        )
        batch: Tuple[torch.Tensor, ...]
        for batch in dataloader:
            input_ids, batch_indices, input_indices, lengths = batch
            outputs = self._ort_session.run(None, {"input_ids": input_ids.numpy()})[0]
            batch_size = input_ids.shape[0]
            for i in range(batch_size):
                length = lengths[i].item()
                batch_idx = batch_indices[i].item()
                input_idx = input_indices[i].item()
                # Remove all batch dims. Remove BOS/EOS from time dim
                segment_ids = input_ids[i, 1 : length - 1].tolist()
                # Shape [B, T]
                probs = outputs[i, 1 : length - 1]
                # Find all positions that exceed the threshold as a sentence boundary
                break_points: List[int] = np.squeeze(np.argwhere(probs > threshold), axis=1).tolist()  # noqa
                # Add the final token to the break points, to not have leftover tokens after the loop
                if (not break_points) or (break_points[-1] != len(segment_ids) - 1):
                    break_points.append(len(segment_ids) - 1)
                # Accumulate results (long inputs are broken into separate elements)
                collectors[batch_idx].collect(
                    ids=segment_ids, break_points=break_points, idx=input_idx,
                )
        outputs: Union[List[str], List[List[str]]] = [x.produce() for x in collectors]
        return outputs
