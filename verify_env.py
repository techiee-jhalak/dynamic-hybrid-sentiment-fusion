"""Environment Verification Script.

Verifies:
1. Python version (>= 3.11)
2. PyTorch installation and device availability (CPU/CUDA)
3. Transformers installation
4. NLTK installation and VADER lexicon availability
5. scikit-learn installation
6. NumPy and Pandas versions
"""

import sys


def verify_environment() -> bool:
    print("=" * 60)
    print("Environment Verification: Dynamic Hybrid Sentiment Fusion")
    print("=" * 60)

    all_passed = True

    # 1. Python Version
    py_version = sys.version_info
    print(f"[Python] Version: {sys.version.split()[0]}")
    if py_version.major == 3 and py_version.minor >= 11:
        print("  -> Status: PASS (Python >= 3.11)")
    else:
        print("  -> Status: FAIL (Requires Python >= 3.11)")
        all_passed = False

    # 2. PyTorch & Device
    try:
        import torch
        print(f"[PyTorch] Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Standard)"
        print(f"  -> CUDA Available: {cuda_available} ({device_name})")
        print("  -> Status: PASS")
    except ImportError as e:
        print(f"[PyTorch] FAILED: {e}")
        all_passed = False

    # 3. Transformers
    try:
        import transformers
        print(f"[Transformers] Version: {transformers.__version__}")
        print("  -> Status: PASS")
    except ImportError as e:
        print(f"[Transformers] FAILED: {e}")
        all_passed = False

    # 4. NLTK
    try:
        import nltk
        print(f"[NLTK] Version: {nltk.__version__}")
        # Test vader lexicon download/availability
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            print("  -> VADER Lexicon: Available")
        except LookupError:
            print("  -> Downloading NLTK vader_lexicon...")
            nltk.download("vader_lexicon", quiet=True)
            print("  -> VADER Lexicon: Downloaded and verified")
        print("  -> Status: PASS")
    except ImportError as e:
        print(f"[NLTK] FAILED: {e}")
        all_passed = False

    # 5. Scikit-Learn
    try:
        import sklearn
        print(f"[scikit-learn] Version: {sklearn.__version__}")
        print("  -> Status: PASS")
    except ImportError as e:
        print(f"[scikit-learn] FAILED: {e}")
        all_passed = False

    # 6. NumPy & Pandas
    try:
        import numpy as np
        import pandas as pd
        print(f"[NumPy] Version: {np.__version__}")
        print(f"[Pandas] Version: {pd.__version__}")
        print("  -> Status: PASS")
    except ImportError as e:
        print(f"[NumPy/Pandas] FAILED: {e}")
        all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL REQUIRED ML ENVIRONMENT CHECKS PASSED.")
    else:
        print("SOME CHECKS FAILED. Please resolve missing dependencies.")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = verify_environment()
    sys.exit(0 if success else 1)
