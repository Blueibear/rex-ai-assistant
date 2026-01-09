"""Quick test to verify transformers compatibility patch works."""

print("Testing transformers BeamSearchScorer patch...")
print("=" * 70)

# Apply the patch - SAME sequence as voice_loop.py
print("\n1. Applying compatibility shim...")
from rex.compat import ensure_transformers_compatibility
ensure_transformers_compatibility()

# Force transformers to load
print("2. Importing transformers...")
import transformers
print(f"   transformers version: {transformers.__version__}")

# Check if BeamSearchScorer is available in transformers namespace
print("\n3. Checking for BeamSearchScorer in transformers namespace...")
if hasattr(transformers, 'BeamSearchScorer'):
    print(f"   ✓ transformers.BeamSearchScorer found: {transformers.BeamSearchScorer}")
else:
    print("   ✗ transformers.BeamSearchScorer NOT found")
    print(f"   Beam-related attributes: {[attr for attr in dir(transformers) if 'Beam' in attr]}")

# Try the import that TTS uses
print("\n4. Testing import statement that TTS uses...")
try:
    from transformers import BeamSearchScorer
    print(f"   ✓ SUCCESS: 'from transformers import BeamSearchScorer' worked!")
    print(f"   BeamSearchScorer class: {BeamSearchScorer}")
    print(f"   Module: {BeamSearchScorer.__module__}")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    print("\n   Trying to find where it actually is...")

    # Try different locations
    locations = [
        "transformers.generation.beam_search",
        "transformers.generation",
        "transformers.generation_utils"
    ]

    for loc in locations:
        try:
            parts = loc.split(".")
            module = __import__(loc, fromlist=[parts[-1]])
            if hasattr(module, 'BeamSearchScorer'):
                print(f"   Found at: {loc}.BeamSearchScorer")
        except (ImportError, AttributeError):
            pass

print("\n" + "=" * 70)
print("Test complete!")
print("\nIf you see '✓ SUCCESS' above, the patch is working correctly.")
print("You can now run: python gui.py")
