# Papers I read

- [Conference rankings](http://portal.core.edu.au/conf-ranks/)
- [Journal rankings](https://www.scimagojr.com/)
- [ris2bib](https://www.bruot.org/ris2bib/)

## [Generating Images with Multimodal Language Models](https://jykoh.com/gill)

Date: 2023-10-13

Conference: [NeurIPS, 2023](https://neurips.cc/virtual/2023/papers.html)

### Proposed

Introduce new mode GILL (Generating Images with Large Language Models) with wide
suite of multimodal capabilities: image retrieval, novel image generation, and
multimodal dialogue.

### What news

First approach capable of conditioning on arbitrarily interleaved image and text
inputs to generate coherent image (and text) outputs.

- Multimodal: It can process both text and images, unlike most text-to-image
  models. This allows for:
  - Image retrieval: Find existing images matching a text description.
  - Image generation: Create novel images based on text.
  - Interleaved text-image generation: Generate text and images together in a
    story-like format.
- Leverages LLM strength: Uses a pre-trained LLM (large language model) for
  strong text understanding, enabling it to handle complex and long
  descriptions.
- Novelty: Creates truly new images, not just retrieving existing ones.
- Flexibility: Decides whether to retrieve or generate an image based on the
  input and context.

### Method

- Learning to process images
- Learning to produce images
- Decide to generate (new) or retrive (already exist) image

### Dataset

- Conceptual Captions 3M (CC3M) is a new dataset consisting of ~3.3M images
  annotated with captions.

### References

- [Paper](https://arxiv.org/pdf/2305.17216.pdf)
- [Code](https://github.com/kohjingyu/gill)
