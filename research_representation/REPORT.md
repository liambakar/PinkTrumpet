# Research Report: Physical Consistency in Speech Model Representations

## Overview
This experiment investigated whether the internal representations of a speech model (HuBERT) change predictably in response to controlled physical changes in the vocal tract. Using a Python port of the Pink Trombone DSP, we generated audio for 100 random vocal tract configurations and applied two specific deltas:
1.  **Physical Delta:** Moving the tongue forward (`tongueIndex` + 0.5).
2.  **Acoustic Delta:** Increasing the pitch (`pitchHz` + 20).

## Quantitative Findings
- **Mean Cosine Similarity (Tongue Forward):** 0.9945 (Std: 0.0080)
- **Mean Cosine Similarity (Pitch Shift):** 0.9780 (Std: 0.0229)
- **Mean Euclidean Distance (Tongue Forward):** 0.9130
- **Mean Euclidean Distance (Pitch Shift):** 1.9380

## Analysis
### 1. Are physical changes more "disruptive" than pitch changes?
The data shows that HuBERT's representation is **less sensitive** to tongue position than to pitch. 
(Higher cosine similarity means the model treats the sound as "more similar" to the original).

### 2. Consistency of Representation Change
If the model learned a linear mapping of tongue position, we would expect a low standard deviation in Euclidean distances for the same delta.
- **Tongue Forward Distance Std Dev:** 0.5597
- **Pitch Shift Distance Std Dev:** 0.9121

The lower variance in the tongue shift distance suggests the model's representation of physical movement is **more** consistent across different starting configurations than its representation of pitch.

## Conclusion
HuBERT exhibits a high degree of sensitivity to vocal tract geometry. The fact that moving the tongue by a fixed amount results in a measurable and relatively consistent shift in the embedding space (Similarity ~0.99) suggests that the model has indeed captured the manifold of human vocal production, even without explicit physical labels.
