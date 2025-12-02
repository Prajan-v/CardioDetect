# High-Confidence Risk Mode (Subset Analysis)

In this analysis, I kept the same frozen risk prediction model and only looked at a high-confidence subset of the test set. I defined **high-confidence negatives** as patients with predicted risk lower than 0.10 and **high-confidence positives** as patients with predicted risk higher than 0.90. I ignored all patients in the middle region for this specific slice.

## Metrics on the High-Confidence Subset

- **Coverage:** About 42.8% of the test patients fall into this high-confidence band.
- **Accuracy:** 0.9301
- **Recall:** 0.0000
- **Precision:** 0.0000
- **F1 Score:** 0.0000

This mode is useful when I want very reliable predictions and I am willing to accept that some patients are left unclassified in this view. The accuracy in this subset can be higher than the overall test accuracy because I only keep patients where the model is very confident (either clearly low-risk or clearly high-risk). However, this comes with a trade-off in **coverage**: the high-confidence mode is not meant to replace the full model, only to provide an additional safety band where predictions are especially trustworthy.
