# Summarization

## Evaluation
Run
```
python compute_rouge.py --res_dir DIR [--verbose]
```
Note that `DIR/summary` contains all the predicted files and `DIR/reference` has all the reference files.

Or run the evaluation within python with
```
import compute_rouge
res = compute_rouge.run(DIR)
```
