from dataclasses import dataclass
from typing import List, Optional, Dict, Union

from sentencepiece import SentencePieceProcessor

# Probably need to move somewhere common
ACRONYM_TOKEN = "<ACRONYM>"


@dataclass
class PCSResultSegment:
    ids: List[int]
    pre_preds: List[Optional[str]]
    post_preds: List[Optional[str]]
    cap_preds: List[List[int]]
    sbd_preds: List[int]

    def __len__(self):
        return len(self.ids)


class PunctCapSegResultCollector:
    def __init__(self, apply_sbd: bool, overlap: int, sp_model: SentencePieceProcessor):
        self._segments: Dict[int, PCSResultSegment] = {}
        self._apply_sbd = apply_sbd
        self._overlap = overlap
        self._sp_model = sp_model

    def collect(
        self,
        ids: List[int],
        pre_preds: List[Optional[str]],
        post_preds: List[Optional[str]],
        sbd_preds: List[int],
        cap_preds: List[List[int]],
        idx: int,
    ):
        self._segments[idx] = PCSResultSegment(
            ids=ids, pre_preds=pre_preds, post_preds=post_preds, sbd_preds=sbd_preds, cap_preds=cap_preds
        )

    def produce(self) -> Union[List[str], str]:
        ids: List[int] = []
        pre_preds: List[Optional[str]] = []
        post_preds: List[Optional[str]] = []
        cap_preds: List[List[int]] = []
        sbd_preds: List[int] = []
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
            pre_preds.extend(segment.pre_preds[start:stop])
            post_preds.extend(segment.post_preds[start:stop])
            sbd_preds.extend(segment.sbd_preds[start:stop])
            cap_preds.extend(segment.cap_preds[start:stop])
        # Apply predictions to input tokens
        input_tokens = [self._sp_model.IdToPiece(x) for x in ids]
        # Segmented sentences
        output_texts: List[str] = []
        # Current sentence, which is built until we hit a sentence boundary prediction
        current_chars: List[str] = []
        for token_idx, token in enumerate(input_tokens):
            # Simple SP decoding
            if token.startswith("▁") and current_chars:
                current_chars.append(" ")
            # Skip non-printable chars
            char_start = 1 if token.startswith("▁") else 0
            for token_char_idx, char in enumerate(token[char_start:], start=char_start):
                # If this is the first char in the subtoken, and we predict "pre-punct", insert it
                if token_char_idx == char_start and pre_preds[token_idx] is not None:
                    current_chars.append(pre_preds[token_idx])
                # If this char should be capitalized, apply upper case
                if cap_preds[token_idx][token_char_idx]:
                    char = char.upper()
                # Append char after pre-punc and upper-casing, before post-punt
                current_chars.append(char)
                # Check if we need to insert "post" punctuation
                label = post_preds[token_idx]
                if label == ACRONYM_TOKEN:
                    # All characters in this subtoken are punctuated with a period
                    current_chars.append(".")
                elif token_char_idx == len(token) - 1 and post_preds[token_idx] is not None:
                    current_chars.append(post_preds[token_idx])
                # If this token is a sentence boundary, finalize the current sentence and reset
                if self._apply_sbd and token_char_idx == len(token) - 1 and sbd_preds[token_idx]:
                    output_texts.append("".join(current_chars))
                    current_chars = []
        if current_chars:
            output_texts.append("".join(current_chars))
        if not self._apply_sbd:
            if len(output_texts) > 1:
                raise ValueError(f"Not applying SBD but got more than one result: {output_texts}")
            return output_texts[0]
        return output_texts
