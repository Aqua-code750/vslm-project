import json
import os
import random

# Curated High-Density Instruction Dataset for Mog1 AI SFT & Pretraining
INSTRUCTION_DATA = [
    # --- Greetings & Conversational Personas ---
    {
        "instruction": "Hello!",
        "response": "Hello! I am Mog1 AI, a Small Language Model. How can I assist you with coding, math, reasoning, or general questions today?"
    },
    {
        "instruction": "Who are you?",
        "response": "I am Mog1 AI (VSLM), an efficient Decoder-Only Transformer language model built in PyTorch by Aqua-code750 and Aquaholograph2014."
    },
    {
        "instruction": "How are you doing?",
        "response": "I am operating smoothly and ready to assist you! What problem or project would you like to explore?"
    },
    {
        "instruction": "What can you help me with?",
        "response": "I can help you write and debug code, solve mathematics and algorithmic problems, explain computer science and machine learning concepts, structure documents, and answer factual questions."
    },
    {
        "instruction": "Good morning!",
        "response": "Good morning! I hope you have a productive day. What can I help you build or learn today?"
    },
    {
        "instruction": "Good night!",
        "response": "Good night! Rest well, and feel free to return whenever you have more questions or ideas."
    },
    {
        "instruction": "Thank you for the explanation!",
        "response": "You are very welcome! Let me know if you would like more details or have follow-up questions."
    },

    # --- Uncertainty & Honest Boundaries (Non-Hallucination) ---
    {
        "instruction": "What will happen in the year 3000?",
        "response": "I cannot predict the exact events of the year 3000, as future historical and scientific developments remain uncertain."
    },
    {
        "instruction": "What is my private password?",
        "response": "I do not know or have access to any personal or private user passwords."
    },
    {
        "instruction": "Can you travel in physical space?",
        "response": "No, I am a software-based neural network model without physical embodiment."
    },

    # --- Mathematics & Step-by-Step Logic ---
    {
        "instruction": "Calculate 15 * 14 step by step.",
        "response": "To compute 15 * 14:\n1. 15 * 10 = 150\n2. 15 * 4 = 60\n3. 150 + 60 = 210\nTherefore, 15 * 14 = 210."
    },
    {
        "instruction": "What is the square root of 144?",
        "response": "The square root of 144 is 12, because 12 * 12 = 144."
    },
    {
        "instruction": "Solve for x in 3x + 9 = 24.",
        "response": "To solve 3x + 9 = 24:\n1. Subtract 9 from both sides: 3x = 15\n2. Divide both sides by 3: x = 5\nResult: x = 5."
    },
    {
        "instruction": "What is 2 raised to the power of 8?",
        "response": "2^8 equals 256."
    },
    {
        "instruction": "What is a prime number?",
        "response": "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself (e.g., 2, 3, 5, 7, 11)."
    },
    {
        "instruction": "Explain the Pythagorean theorem.",
        "response": "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse (c) is equal to the sum of the squares of the other two sides (a and b): a^2 + b^2 = c^2."
    },

    # --- Computer Science & Data Structures ---
    {
        "instruction": "What is Big O notation?",
        "response": "Big O notation mathematically describes the upper bound of an algorithm's time or space complexity as input size n grows asymptotically."
    },
    {
        "instruction": "Explain Binary Search and its time complexity.",
        "response": "Binary search finds a target element in a sorted array by repeatedly dividing the search range in half. It has an average and worst-case time complexity of O(log n)."
    },
    {
        "instruction": "What is the difference between a Stack and a Queue?",
        "response": "A Stack operates on a Last-In, First-Out (LIFO) order where items are added and removed from the top. A Queue operates on a First-In, First-Out (FIFO) order where items enter at the rear and exit from the front."
    },
    {
        "instruction": "How does a Hash Table achieve O(1) average lookup?",
        "response": "A hash table uses a hash function to convert a key into an integer index in an underlying array, allowing direct memory access in O(1) average time."
    },
    {
        "instruction": "Explain Quicksort.",
        "response": "Quicksort is a divide-and-conquer sorting algorithm. It selects a pivot element, partitions the array into elements smaller and larger than the pivot, and recursively sorts the sub-arrays with O(n log n) average time complexity."
    },
    {
        "instruction": "What is Dynamic Programming?",
        "response": "Dynamic Programming is an optimization technique that solves complex problems by breaking them into overlapping subproblems and storing the computed results (memoization or tabulation) to prevent redundant work."
    },
    {
        "instruction": "What is a Binary Search Tree (BST)?",
        "response": "A Binary Search Tree is a binary tree where for each node, all values in its left subtree are smaller than the node's value, and all values in its right subtree are greater."
    },

    # --- Python Programming & Software Engineering ---
    {
        "instruction": "Write a Python function to check if a string is a palindrome.",
        "response": "```python\ndef is_palindrome(s: str) -> bool:\n    clean = ''.join(c.lower() for c in s if c.isalnum())\n    return clean == clean[::-1]\n\n# Example usage:\nprint(is_palindrome('racecar'))  # True\nprint(is_palindrome('hello'))    # False\n```"
    },
    {
        "instruction": "Write a Python function to calculate the factorial of a number.",
        "response": "```python\ndef factorial(n: int) -> int:\n    if n < 0:\n        raise ValueError('Factorial is not defined for negative numbers.')\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n\n# Example:\nprint(factorial(5))  # Output: 120\n```"
    },
    {
        "instruction": "Write a Python function for binary search.",
        "response": "```python\ndef binary_search(arr: list, target: int) -> int:\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```"
    },
    {
        "instruction": "Explain Python list comprehensions with an example.",
        "response": "List comprehension provides a concise syntax to create lists. Example:\n```python\n# Create a list of squares for even numbers\nsquares = [x**2 for x in range(10) if x % 2 == 0]\nprint(squares)  # [0, 4, 16, 36, 64]\n```"
    },
    {
        "instruction": "What is the difference between a list and a tuple in Python?",
        "response": "In Python, lists are mutable (elements can be modified, appended, or removed) and defined using square brackets `[]`. Tuples are immutable (cannot be changed after creation) and defined using parentheses `()`."
    },
    {
        "instruction": "What is Git and why is it used?",
        "response": "Git is a distributed version control system that tracks modifications in source code files, supports non-linear branching and merging, and enables collaborative software development."
    },
    {
        "instruction": "What is an API?",
        "response": "An Application Programming Interface (API) is a formal specification that allows distinct software applications to communicate and transfer data using standard protocols like HTTP and JSON."
    },
    {
        "instruction": "What is Docker?",
        "response": "Docker is a containerization platform that packages an application and all its system dependencies together into an isolated, lightweight container that runs consistently across any operating environment."
    },

    # --- Transformer & Neural Network Architecture ---
    {
        "instruction": "What is Self-Attention in Transformers?",
        "response": "Self-attention computes dynamic attention weights between all token pairs in a sequence using Query (Q), Key (K), and Value (V) projections: Attention(Q, K, V) = softmax((Q @ K.T) / sqrt(d_k)) @ V."
    },
    {
        "instruction": "What is Rotary Position Embedding (RoPE)?",
        "response": "Rotary Position Embedding (RoPE) encodes token positions by multiplying query and key vectors by rotation matrices. It incorporates relative positional distance directly into the dot-product attention score."
    },
    {
        "instruction": "What is RMSNorm and why is it preferred over LayerNorm?",
        "response": "RMSNorm normalizes layer inputs by their root mean square value without subtracting the mean. It reduces computational overhead while maintaining gradient training stability in modern LLMs like LLaMA and Mog1."
    },
    {
        "instruction": "What is SwiGLU?",
        "response": "SwiGLU (Swish-Gated Linear Unit) is an MLP activation architecture defined as SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down. It provides enhanced non-linear representation capacity over standard GELU."
    },
    {
        "instruction": "Why is Weight Tying used in language models?",
        "response": "Weight tying shares the parameter matrix between the input token embedding layer and the final output language model head. It significantly reduces parameter count and improves regularization on small vocabularies."
    },
    {
        "instruction": "What is a Key-Value (KV) Cache in LLM inference?",
        "response": "A KV Cache stores precomputed Key and Value attention tensors from preceding generation steps. During autoregressive decoding, the model only computes attention for the newest token, reducing per-token complexity from O(T^2) to O(T)."
    },
    {
        "instruction": "What is AdamW optimizer?",
        "response": "AdamW is an adaptive learning rate optimizer that correctly decouples L2 weight decay from moving gradient updates, preventing weight decay decay degradation observed in standard Adam."
    },
    {
        "instruction": "What is Gradient Clipping?",
        "response": "Gradient clipping caps the global norm of parameter gradients to a maximum threshold (e.g. 1.0) during backpropagation, preventing numerical instability and exploding gradients."
    },

    # --- Structured Formatting Instructions ---
    {
        "instruction": "List three advantages of Small Language Models in bullet points.",
        "response": "• **Low Latency**: Fast inference speeds on commodity CPU and edge hardware.\n• **Low Memory Footprint**: Fits inside memory-constrained devices without requiring multi-GPU clusters.\n• **Cost Efficiency**: Economical to train, fine-tune, and deploy continuously."
    },
    {
        "instruction": "Provide a valid JSON object describing a book with title, author, and year.",
        "response": "```json\n{\n  \"title\": \"Structure and Interpretation of Computer Programs\",\n  \"author\": \"Harold Abelson and Gerald Jay Sussman\",\n  \"year\": 1985\n}\n```"
    },
    {
        "instruction": "Format a comparison table between CPU and GPU.",
        "response": "| Feature | CPU | GPU |\n| :--- | :--- | :--- |\n| **Core Architecture** | Few powerful, complex cores | Thousands of smaller, efficient cores |\n| **Workload Specialty** | Sequential logic & single-thread performance | Massive parallel computation (Matrix multiplication) |\n| **Primary Use** | Operating systems, general applications | Deep learning training, rendering, gaming |"
    },

    # --- General Science & Physics ---
    {
        "instruction": "What is gravity?",
        "response": "Gravity is a fundamental natural interaction by which all entities with mass or energy attract one another. In Einstein's General Relativity, gravity is described as the curvature of spacetime caused by mass."
    },
    {
        "instruction": "What is photosynthesis?",
        "response": "Photosynthesis is the biological process used by plants, algae, and cyanobacteria to convert solar light energy, water (H2O), and carbon dioxide (CO2) into oxygen (O2) and chemical glucose energy."
    },
    {
        "instruction": "What is DNA?",
        "response": "Deoxyribonucleic Acid (DNA) is the double-helix molecule that contains the genetic instructions necessary for the growth, development, functioning, and reproduction of all known living organisms."
    },
    {
        "instruction": "What is the speed of light?",
        "response": "The speed of light in a vacuum is exactly 299,792,458 meters per second (approximately 300,000 km/s)."
    },
    {
        "instruction": "What is an atom?",
        "response": "An atom is the fundamental building block of chemical elements, composed of a dense nucleus of protons and neutrons surrounded by a cloud of negatively charged electrons."
    }
]

def generate_datasets(save_dir: str = "data", train_ratio: float = 0.9):
    os.makedirs(save_dir, exist_ok=True)
    
    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    data = list(INSTRUCTION_DATA)
    random.shuffle(data)
    
    split_idx = int(len(data) * train_ratio)
    train_set = data[:split_idx]
    val_set = data[split_idx:]
    
    train_file = os.path.join(save_dir, "train_instructions.json")
    val_file = os.path.join(save_dir, "val_instructions.json")
    
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_set, f, indent=2, ensure_ascii=False)
        
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_set, f, indent=2, ensure_ascii=False)
        
    print(f"Generated datasets: {len(train_set)} training samples -> '{train_file}', {len(val_set)} validation samples -> '{val_file}'.")
    return train_file, val_file

if __name__ == "__main__":
    generate_datasets()
