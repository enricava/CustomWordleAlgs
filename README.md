# How to run the simulations:

1. Download `allowed_words.txt` and `solutions.txt`.
2. Download and run `efficient_precomputation.py` in the same folder to generate `datafile.npy`.
3. Download and run `priority_precomputation.py` in the same folder to generate `word_priority.npy`.
4. Download and run the latest version of `Simulator.py`.

# Requirements

Python 3 with `tqdm` library for progress bars.

To install `tqdm` open a command prompt window and type:
`pip install tqdm`

# About

 ### efficient_precomputation.py
>Precomputes a file with ternary wordle patterns resulting after cross examining words from a .txt file using vectorization.
>This will generate and store two numpy matrices:

  `matrix1[#word1, #word2]` := pattern result after submitting word1 when solution is word2
  
  `matrix2[#word]` := word entropy

### old_prec.py
-- DO NOT USE ANY MORE, for precomputation use efficient_precomputation.py

>Precomputes a file with ternary wordle patterns resulting after cross examining words from a .txt file.
>This will generate and store two numpy matrices:

  `matrix1[#word1, #word2]` := pattern result after submitting word2 when solution is word1
  
  `matrix2[#word, pattern]` := frequency of pattern given submitted word. For example purposes only
  
 ### allowed_words.txt
 All submittable wordle words. Extracted from [3b1b](https://github.com/3b1b/videos/tree/master/_2022)
 
 ### solutions.txt
 All possible wordle solutions. Extracted from [3b1b](https://github.com/3b1b/videos/tree/master/_2022)
