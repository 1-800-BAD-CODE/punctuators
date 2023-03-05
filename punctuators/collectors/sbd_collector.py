from dataclasses import dataclass
from typing import List, Dict

from sentencepiece import SentencePieceProcessor


@dataclass
class SBDResultSegment:
    ids: List[int]
    break_points: List[int]

    def __len__(self):
        return len(self.ids)


class SBDResultCollector:
    def __init__(self, overlap: int, sp_model: SentencePieceProcessor):
        self._segments: Dict[int, SBDResultSegment] = {}
        self._overlap = overlap
        self._sp_model = sp_model

    def collect(
        self, ids: List[int], break_points: List[int], idx: int,
    ):
        self._segments[idx] = SBDResultSegment(ids=ids, break_points=break_points)

    def produce(self) -> List[str]:
        ids: List[int] = []
        break_points: List[int] = []
        for i in range(len(self._segments)):
            segment = self._segments[i]
            # Assume BOS and EOS stripped already
            start = 0
            stop = len(segment)
            # If there is a segment before this one, drop the beginning overlap
            if i > 0:
                start += self._overlap // 2
            # If there is a segment after this one, drop the end overlap
            if i < len(self._segments) - 1:
                stop -= self._overlap // 2
            ids.extend(segment.ids[start:stop])
            break_points.extend(segment.break_points[start:stop])
        out_texts: List[str] = []
        for i, break_point in enumerate(break_points):
            start = 0 if i == 0 else (break_points[i - 1] + 1)
            sub_ids = ids[start : break_point + 1]
            sub_text = self._sp_model.DecodeIds(sub_ids)
            out_texts.append(sub_text)
        return out_texts
