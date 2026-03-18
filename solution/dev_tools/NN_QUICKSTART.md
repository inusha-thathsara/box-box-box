## Neural Network Quick Start Guide

### What This Does
Trains a small 2-layer neural network to predict driver ranking scores from 44-dimensional feature vectors. The NN learns nonlinear patterns in pit strategy, tire degradation, and driver characteristics that the physics model might miss.

### Architecture
- **Input:** 44-dimensional feature vector (tire config, pit timing, driver ID, race setup)
- **Hidden layers:** 128 → 64 neurons with ReLU activation
- **Output:** Single score value (higher = better driver/strategy)
- **Loss function:** Pairwise margin loss (ensures driver A ranking higher than driver B in actual finish order gets score_A > score_B)

### Prerequisites
Requires PyTorch. Install with:
```bash
pip install torch
```

### Step 1: Train the Model
From the solution/ folder:

**Option A: CPU training (10-20 minutes)**
```bash
python train_nn_model.py --epochs 10 --lr 0.001 --batch-pairs 500 --output nn_model.pth
```

**Option B: GPU training (faster, requires CUDA)**
```bash
python train_nn_model.py --epochs 10 --lr 0.001 --batch-pairs 500 --device cuda --output nn_model.pth
```

**Options B: Extended training (more epochs for potential better results)**
```bash
python train_nn_model.py --epochs 20 --lr 0.001 --batch-pairs 1000 --output nn_model.pth
```

### Step 2: Test the Model
From root folder (box-box-box):

**PowerShell:**
```powershell
cd solution
python test_nn_model.ps1
```

**Python (manual A/B test):**
```python
import subprocess
import json

# Test with NN
import os
os.environ['BOXBOXBOX_USE_NN'] = '1'

passed = 0
for i in range(1, 101):
    test_num = f'{i:03d}'
    input_file = f'data/test_cases/inputs/test_{test_num}.json'
    expected_file = f'data/test_cases/expected_outputs/test_{test_num}.json'
    
    with open(input_file) as f:
        test_input = json.load(f)
    with open(expected_file) as f:
        expected = json.load(f)
    
    # Run simulator
    result = subprocess.run(['python', 'solution/race_simulator.py'], 
                          input=json.dumps(test_input), 
                          capture_output=True, 
                          text=True)
    output = json.loads(result.stdout)
    
    # Check if exact match
    if output['finishing_positions'] == expected['finishing_positions']:
        passed += 1

print(f"NN Accuracy: {passed}%")
```

### Step 3: Enable for Submission
The race simulator automatically detects and uses `nn_model.pth` if the environment variable is set:

```bash
export BOXBOXBOX_USE_NN=1
# Windows PowerShell:
$env:BOXBOXBOX_USE_NN = '1'
```

Or to always use NN, the simulator will pick it up if `solution/nn_model.pth` exists (enabled via env var).

### Training Parameters Explanation

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--epochs` | 10 | More epochs = longer training but potentially better results. Try 15-20 for extended run. |
| `--lr` | 0.001 | Learning rate. Lower (0.0001-0.0005) = slower but more stable. Higher (0.005) = faster but may diverge. |
| `--batch-pairs` | 500 | Pairs per race to sample for training. Higher = more diverse training but slower per epoch. |
| `--device` | cpu | Use 'cuda' for GPU (must have PyTorch CUDA version installed) |
| `--output` | nn_model.pth | Output file path |

### Expected Results
- Baseline (physics only): ~9%
- NN Model (typical): 9-15% accuracy
  - Best case (if significant nonlinear patterns exist): 15-20%
  - Worst case (if feature set insufficient): 9-12%
  
**Reality check:** The 91% failure rate is likely due to missing features (driver skill, team effects, specific setup parameters). NN can help with ~1-6% improvement if nonlinear patterns exist.

### Troubleshooting

**PyTorch not found:**
```bash
pip install torch
```

**Model file not loading:**
- Check `solution/nn_model.pth` exists
- Check permissions on file
- Try deleting and retraining

**Low accuracy (≤9%):**
- NN may have failed to learn. Try:
  - More epochs: `--epochs 20`
  - Different learning rate: `--lr 0.0005` or `--lr 0.002`
  - Longer training: `--batch-pairs 1000`

**Training too slow:**
- Install GPU support (CUDA 11.8+ required):
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```
- Use `--device cuda`
- Reduce batch-pairs: `--batch-pairs 250` (but results may suffer)

### Files Created
- `train_nn_model.py` - Training script
- `nn_model.pth` - Trained model (created after training)
- `test_nn_model.ps1` - A/B testing script
- (Modified) `race_simulator.py` - Added NN inference functions

### Next Steps
1. **Install PyTorch:** `pip install torch`
2. **Run training:** `python solution/train_nn_model.py --epochs 10`
3. **Test accuracy:** Run test_nn_model.ps1 (or A/B test manually)
4. **If NN improves,' accuracy, submit with BOXBOXBOX_USE_NN='1'
5. **If not,** stick with baseline physics model
