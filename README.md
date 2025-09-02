# MGR-CSC repository
Multilingual Generative Retrieval via Cross-lingual Semantic Compression.

**Abstract:** Generative Information Retrieval is an emerging retrieval paradigm that exhibits remarkable performance in monolingual scenarios.
However, applying these methods to multilingual retrieval still encounters two primary challenges, cross-lingual identifier misalignment and identifier inflation. 
To address these limitations, we propose Multilingual Generative Retrieval via Cross-lingual Semantic Compression (MGR-CSC), a novel framework that unifies semantically equivalent multilingual keywords into shared atoms to align semantics and compresses the identifier space, and we propose a dynamic multi-step constrained decoding strategy during retrieval. 
MGR-CSC improves cross-lingual alignment by assigning consistent identifiers and enhances decoding efficiency by reducing redundancy. 
Experiments demonstrate that MGR-CSC achieves outstanding retrieval accuracy, improving by $6.83\%$ on mMarco100k and $4.77\%$ on mNQ320k, while reducing document identifiers length by $74.51\%$ and $78.2\%$, respectively.

## Description
We have open-sourced the implementation of MGR-CSC and the dataset mNQ320k we constructed, which can be used to reproduce the results in the paper.

> We upload the full [mNQ320k dataset](https://huggingface.co/datasets/sssssimeng/mNQ320k) now.

## Installation
	> pip install -r requirement.txt 

## Getting Started

	> sh run.sh 





