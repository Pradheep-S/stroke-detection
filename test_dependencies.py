#!/usr/bin/env python
"""Quick test script to verify dependencies"""
import sys
print("Testing dependencies...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
except ImportError as e:
    print(f"✗ TensorFlow: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas: {e}")
    sys.exit(1)

try:
    import sklearn
    print(f"✓ Scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"✗ Scikit-learn: {e}")
    sys.exit(1)

try:
    import imblearn
    print("✓ Imbalanced-learn")
except ImportError as e:
    print(f"✗ Imbalanced-learn: {e}")
    sys.exit(1)

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib: {e}")
    sys.exit(1)

print("\n✓ All dependencies ready!")
print("\nNow run: python main.py")
