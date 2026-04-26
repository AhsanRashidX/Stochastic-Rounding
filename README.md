your_project/
├── gpt_experiment.py          # Your main file (modified)
├── quantization_utils.py       # NEW: Quantization functions
├── quantized_layers.py         # NEW: Custom quantized layers
└── train_utils.py              # NEW: Training utilities

The experimental results reveal several important trends regarding low-precision training. First, stochastic rounding at 8-bit precision achieves performance nearly identical to the FP32 baseline, confirming that reduced precision does not necessarily compromise model quality. In contrast, round-to-nearest (RTN) consistently performs slightly worse, which can be attributed to its inherent bias .


run this command:                                    
python gpt_experiment.py
