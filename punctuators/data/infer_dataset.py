from dataclasses import dataclass
from typing import Iterator, List, Iterable, Tuple

import torch
from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset


@dataclass
class TokenizedSegment:
    input_ids: List[int]
    batch_idx: int
    input_idx: int

    def __len__(self):
        return len(self.input_ids)


class TokenBatchSampler(Iterable):
    def __init__(self, segments: List[TokenizedSegment], batch_size_tokens: int):
        super().__init__()
        # Make a list of tuples [tokens, batch_idx] where each `tokens`
        self._batches: List[List[int]] = self._make_batches(segments=segments, batch_size_tokens=batch_size_tokens)

    def _make_batches(self, segments: List[TokenizedSegment], batch_size_tokens: int) -> List[List[int]]:
        segments_with_index: List[Tuple[TokenizedSegment, int]] = [(segment, i) for i, segment in enumerate(segments)]
        segments_with_index = sorted(segments_with_index, key=lambda x: len(x[0]), reverse=True)
        # All batches
        batches: List[List[int]] = []
        # Max length in the batch
        current_max_len: int = 0
        # Next batch
        current_batch_elements: List[int] = []
        for segment, idx in segments_with_index:
            # Check if this segment would cause the batch to exceed num tokens, including padding
            potential_max_len = max(current_max_len, len(segment))
            if potential_max_len * (len(current_batch_elements) + 1) > batch_size_tokens:
                batches.append(current_batch_elements)
                current_batch_elements = []
                current_max_len = 0
            current_batch_elements.append(idx)
            current_max_len = max(current_max_len, len(segment))
        # Check for final partial batch
        if current_batch_elements:
            batches.append(current_batch_elements)
        return batches

    def __iter__(self) -> Iterator:
        for batch in self._batches:
            yield batch

    def __len__(self):
        return len(self._batches)


class TextInferenceDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        spe_model_path: str,
        batch_size_tokens: int = 4096,
        max_length: int = 512,
        overlap: int = 32,
    ):
        super().__init__()
        self._spe_model: SentencePieceProcessor = SentencePieceProcessor(spe_model_path)  # noqa
        self._segments: List[TokenizedSegment] = self._tokenize_inputs(texts=texts, max_len=max_length, overlap=overlap)
        self._sampler = TokenBatchSampler(segments=self._segments, batch_size_tokens=batch_size_tokens)

    @property
    def sampler(self) -> Iterable:
        return self._sampler

    def _tokenize_inputs(self, texts: List[str], max_len: int, overlap: int) -> List[TokenizedSegment]:
        # To leave room for EOS + BOS, use 2 less than max length
        max_len = max_len - 2
        segments: List[TokenizedSegment] = []
        for batch_idx, text in enumerate(texts):
            ids = self._spe_model.EncodeAsIds(text)
            start = 0
            input_idx = 0
            while start < len(ids):
                adjusted_start = start - (0 if input_idx == 0 else overlap)
                stop = adjusted_start + max_len
                segments.append(
                    TokenizedSegment(input_ids=ids[adjusted_start:stop], input_idx=input_idx, batch_idx=batch_idx)
                )
                start = stop
                input_idx += 1
        return segments

    def __len__(self):
        return len(self._segments)

    def __getitem__(self, idx):
        """


        Returns:
             Tuple (tokens, batch_idx, input_idx)
        """
        segment: TokenizedSegment = self._segments[idx]
        input_ids = torch.Tensor([self._spe_model.bos_id()] + segment.input_ids + [self._spe_model.eos_id()])
        return input_ids, segment.batch_idx, segment.input_idx

    def collate_fn(self, batch):
        input_ids: List[torch.Tensor] = [x[0] for x in batch]
        batch_indices: torch.Tensor = torch.tensor([x[1] for x in batch])
        input_indices: torch.Tensor = torch.tensor([x[2] for x in batch])
        lengths = torch.tensor([x.shape[0] for x in input_ids])
        batch_size = len(input_ids)
        max_len = lengths.max().item()
        batched_ids: torch.Tensor = torch.full(size=[batch_size, max_len], fill_value=self._spe_model.pad_id())
        for idx, ids in enumerate(input_ids):
            batched_ids[idx, : lengths[idx]] = input_ids[idx]
        return batched_ids, batch_indices, input_indices, lengths
