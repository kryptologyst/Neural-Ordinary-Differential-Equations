#!/usr/bin/env python3
"""Simple test script to verify Neural ODEs installation."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from models import BasicODEFunc, NeuralODE, LinearBaseline
        print("✅ Model imports successful")
        return True
    except ImportError as e:
        print(f"❌ Model import failed: {e}")
        return False

def test_neural_ode():
    """Test basic Neural ODE functionality."""
    try:
        import torch
        from models import BasicODEFunc, NeuralODE
        
        print("Testing Neural ODE...")
        ode_func = BasicODEFunc(hidden_dim=16)
        model = NeuralODE(ode_func=ode_func)
        
        x0 = torch.tensor([[0.0]])
        t = torch.linspace(0, 1, 10)
        output = model(x0, t)
        
        print(f"✅ Input shape: {x0.shape}")
        print(f"✅ Time points: {t.shape}")
        print(f"✅ Output shape: {output.shape}")
        print("✅ Neural ODE test passed!")
        return True
    except Exception as e:
        print(f"❌ Neural ODE test failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    try:
        from data import SyntheticSineDataset
        
        print("Testing data loading...")
        dataset = SyntheticSineDataset(n_samples=50, split="train")
        sample = dataset[0]
        
        print(f"✅ Dataset length: {len(dataset)}")
        print(f"✅ Sample keys: {list(sample.keys())}")
        print("✅ Data loading test passed!")
        return True
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧠 Neural ODEs Installation Test")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Neural ODE Test", test_neural_ode),
        ("Data Loading Test", test_data_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Neural ODEs is ready to use.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
