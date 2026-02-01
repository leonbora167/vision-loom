# vision-loom

[![wakatime](https://wakatime.com/badge/user/23cfbce7-d612-40f6-82c3-807d45099810/project/2e3af9f3-09b2-4f68-bbdd-49ff04799fa2.svg)](https://wakatime.com/badge/user/23cfbce7-d612-40f6-82c3-807d45099810/project/2e3af9f3-09b2-4f68-bbdd-49ff04799fa2)

In the current AI landscape, we face a Latency-Intelligence Trade-off:

* Foundation Models (VLMs/LLMs): Contain massive world knowledge and can identify objects via natural language prompts, but are too computationally expensive for real-time edge inference.

* Classical Models (YOLO, MobileNet, etc.): Highly efficient and capable of 60+ FPS, but require thousands of high-quality, manually labeled images to perform well.

Vision Loom acts as the "Loom" that weaves these two together. It uses models like Grounding DINO and SAM-2 as automated oracles. Instead of a human clicking bounding boxes for weeks, Vision Loom uses the internal "knowledge" of these large models to pseudo-label your custom datasets. This allows you to train a lightweight model that inherits the accuracy of a giant.

## Why should you use vision-loom 

* **Accelerate R&D:** Go from a raw folder of images to a trained YOLO/segmentation model in hours, not months.

* **Prompt-Based Labeling:** Label complex objects (e.g., "the rusted part of a pipe") using text prompts rather than manual drawing.

* **Cost Efficiency:** Reduce the need for expensive labeling outsourcing by using state-of-the-art open-source foundation models.

* **Data Quality:** Ensure consistent labeling logic across thousands of images by defining strict VLM prompts.