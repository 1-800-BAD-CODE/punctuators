# Overview

`punctuators` is a project for inference for punctuation and related analytics.

This project is a mostly-undocumented prototype at the moment.

The links to models below contain sufficient documentation for each model.

# Installation

This project can be installed with `pip`:

```bash
$ pip install punctuators
```

# Supported Models

This section lists the models currently supported by this package.

## Punctuation, True-Casing, and Sentence Boundary Detection

These models perform punctuation restoration, true-casing (capitalization), and sentence boundary detection (
segmentation).
These analytics together are referred to as PCS (punctuation, capitalization, segmentation).

### 47-language PCS

The following model card describes a base-sized model that can perform PCS on 47 common languages:
https://huggingface.co/1-800-BAD-CODE/punct_cap_seg_47_language

## Sentence Boundary Detection

Sentence Boundary Detection (SBD) is the simpler task of accepting punctuated input and segmenting the input into
separate sentences.

### 49-language SBD

The following model card describes a small-sized model that can perform SBD on 49 common languages:
https://huggingface.co/1-800-BAD-CODE/sentence_boundary_detection_multilang 

