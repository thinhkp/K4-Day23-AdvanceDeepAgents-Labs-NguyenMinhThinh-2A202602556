# Survey of Video and Multimodal Generation

## TL;DR
- Modern video generation has shifted from frame-based approaches to unified space-time diffusion architectures, enabling higher temporal consistency [1][2].
- Diffusion Transformers (DiTs) have emerged as the dominant backbone, replacing traditional U-Nets to better handle long-range dependencies in latent spaces [3][4].
- Evaluation has evolved from distribution-based metrics like FVD [5] to hybrid, model-based benchmarks that simulate human feedback, such as VIDEOSCORE and Video-Bench [6][7].

## Background
Video generation involves synthesizing coherent sequences of images that maintain temporal, spatial, and semantic consistency. Early methods struggled with temporal flickering and limited motion dynamics. Recent advancements leverage latent diffusion models (LDMs) and transformer architectures to model the complex spatiotemporal distribution of video data.

## Foundational Architectures
The field has seen a transition toward latent space representations. Latte [3] utilizes a latent diffusion transformer to model video as a sequence of patches, effectively capturing global temporal dynamics. ProAV-DiT [4] extends this by integrating synchronized audio-video generation, demonstrating the versatility of transformer-based backbones. Furthermore, world models [8] provide a framework for understanding and predicting future states, which is critical for interactive and long-form generation.

## Recent Advances in Video Synthesis
State-of-the-art models now prioritize unified space-time architectures. Lumiere [1] employs a space-time U-Net to generate full-frame, temporally consistent videos in a single pass. xGen-VideoSyn-1 [2] focuses on high-fidelity synthesis by utilizing compressed representations, reducing computational overhead while maintaining visual quality. VideoFactory [9] introduces "Swap-CA" (Cross-Attention) mechanisms to better align generated content with textual prompts, addressing the challenge of semantic fidelity in long-duration videos.

## Evaluation Metrics and Benchmarks
Traditional metrics like Fréchet Video Distance (FVD) [5] have been the standard for measuring distribution similarity between real and generated videos. However, these metrics often fail to capture fine-grained semantic or temporal nuances. Newer approaches like VIDEOSCORE [7] and Video-Bench [6] utilize Large Multimodal Models (LMMs) to simulate human feedback, providing multi-aspect evaluations that correlate more strongly with human preference.

## Trends and Open Problems
- **Efficiency:** Scaling video generation to higher resolutions and longer durations remains computationally expensive.
- **Controllability:** Achieving precise control over motion, camera movement, and object interaction is an ongoing challenge.
- **Evaluation:** Developing metrics that reliably capture temporal coherence and semantic alignment without relying on expensive human evaluation is a key research priority.

## References
[1] Lumiere: A Space-Time Diffusion Model for Video Generation. web. https://arxiv.org/abs/2401.12945 (2024-02-05)
[2] xGen-VideoSyn-1: High-fidelity Text-to-Video Synthesis with Compressed Representations. web. https://www.arxiv.org/abs/2408.12590 (2024-08-31)
[3] Latte: Latent Diffusion Transformer for Video Generation. hf-search. https://huggingface.co/papers/2401.03048 (2024-01-05)
[4] ProAV-DiT: A Projected Latent Diffusion Transformer for Efficient Synchronized Audio-Video Generation. hf-search. https://huggingface.co/papers/2511.12072 (2025-11-15)
[5] Towards Accurate Generative Models of Video: A New Metric & Challenges. arxiv. https://arxiv.org/abs/1812.01717 (2019-03-27)
[6] Video-Bench: Human-Aligned Video Generation Benchmark. web. https://openaccess.thecvf.com/content/CVPR2025/papers/Han_Video-Bench_Human-Aligned_Video_Generation_Benchmark_CVPR_2025_paper.pdf (n.d.)
[7] VIDEOSCORE: Building Automatic Metrics to Simulate Fine-grained Human Feedback. web. https://aclanthology.org/2024.emnlp-main.127.pdf (n.d.)
[8] World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning Paradigms, and Applications. arxiv. https://arxiv.org/abs/2606.00133 (2026-05-28)
[9] VideoFactory: Swap Attention in Spatiotemporal Diffusions for Text-to-Video Generation. web. https://ar5iv.labs.arxiv.org/html/2305.10874 (n.d.)
