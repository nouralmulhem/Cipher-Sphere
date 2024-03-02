from collections import Counter


from collections import Counter

def count_freq(words, X):
  word_counts = Counter(words)

  sorted_words = sorted(word_counts, key=lambda word: (-word_counts[word], word))

  return sorted_words[:X]


# test lexicographical order
if __name__ == '__main__':
  arr = ["pharaoh", "sphinx", "pharaoh", "pharaoh", "naps", "nile", "sphinx", "pyramid", "pharaoh", "sphinx", "sphinx"]
  X = 4
  print(count_freq(arr, X))