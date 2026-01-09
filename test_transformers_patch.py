"""Quick test to verify transformers compatibility patch works."""

print("Testing transformers BeamSearchScorer patch...")

# Apply the patch
print("1. Applying compatibility shim...")
from rex.compat import ensure_transformers_compatibility
ensure_transformers_compatibility()

# Try to import transformers
print("2. Importing transformers...")
import transformers
print(f"   transformers version: {transformers.__version__}")

# Check if BeamSearchScorer is available
print("3. Checking for BeamSearchScorer...")
if hasattr(transformers, 'BeamSearchScorer'):
    print(f"   ✓ BeamSearchScorer found: {transformers.BeamSearchScorer}")
else:
    print("   ✗ BeamSearchScorer NOT found in transformers namespace")
    print(f"   Available attributes: {[attr for attr in dir(transformers) if 'Beam' in attr]}")

# Try the import that TTS uses
print("4. Testing import as TTS does it...")
try:
    from transformers import BeamSearchScorer
    print(f"   ✓ SUCCESS: from transformers import BeamSearchScorer worked!")
    print(f"   BeamSearchScorer: {BeamSearchScorer}")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    # Try to see where it actually is
    try:
        from transformers.generation import BeamSearchScorer as BSS
        print(f"   But it IS available at: transformers.generation.BeamSearchScorer")
    except ImportError:
        print(f"   And it's NOT in transformers.generation either!")

print("\nDone!")
