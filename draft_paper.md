# Draft paper to read

## [Detecting Deepfakes Without Seeing Any](https://paperswithcode.com/paper/detecting-deepfakes-without-seeing-any)

Date: 2023-11-02

### Why?

Deepfakes pose a threat to society. One person can fake another person data to
scam others, or hacker bypass into another system using fake identity which
raise bigger risk.

Traditional deepfake detection methods struggle with zero-day attacks, meaning
they can only detect fakes they've already seen. This becomes less effective as
new manipulation techniques emerge rapidly.

### Proposed solution

FACTOR, the proposed method, checks if the "claimed facts" about the content
(e.g., who a person is, what they're saying) match the actual observations
within the content. Disparities between claim and reality suggest a deepfake.

### What news

It is training free, easy to implement, it achieves better than state-of-the-art
accuracy,
[see](https://paperswithcode.com/sota/deepfake-detection-on-fakeavceleb-1).

### Dataset

- Celeb-DF is a large-scale dataset of real and deepfake celebrity videos,
  designed to challenge and improve deepfake detection algorithms, was generated
  through face swapping involving 59 pairs of distinct identities.
  - 590 real videos
  - 5639 fake videos
- DFD (DeepFake Detection) dataset is a collection of videos designed to train
  and evaluate deepfake detection models. It's one of the earliest and most
  widely used datasets in this field, and it has contributed significantly to
  the development of deepfake detection techniques.
  - 363 real videos
  - 3068 synthetically generated fake videos
- DFDC (DeepFake Detection Challenge) dataset is a large-scale, publicly
  available dataset designed specifically to advance deepfake detection
  research. It's known for its exceptional size, diversity, and realism, posing
  significant challenges to existing detection methods and driving innovation in
  the field.
  - 1133 real videos
  - 4080 manipulated videos
- The FF++(C23) is a specific version of the FaceForensics++ dataset, a
  collection of videos designed for training and evaluating deepfake detection
  models. It contains over 1,000 videos of real and manipulated faces, created
  using various deepfake generation techniques.

### Coding step

- Extract frame from videos
- Extract Region of Interest (ROIs) and align
- Feature extraction (face representation)
- Evaluation (require claimed facts in `.npy` files)

### Limitation

How to get "claimed facts" about one person correctly? Need authorization,
license, ...?

### References

- [Paper](https://arxiv.org/pdf/2311.01458v1.pdf)
- [Code](https://github.com/talreiss/FACTOR)

## [Generating Images with Multimodal Language Models](https://jykoh.com/gill)

Date: 20231013

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

### Dataset

- Conceptual Captions 3M (CC3M) is a new dataset consisting of ~3.3M images
  annotated with captions.

### References

- [Paper](https://arxiv.org/pdf/2305.17216.pdf)
- [Code](https://github.com/kohjingyu/gill)
