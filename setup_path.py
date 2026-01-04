"""
Add the scaledown directory to Python path.
Import this at the top of any script to use scaledown modules.
"""

import sys
from pathlib import Path

# Add the soft_compression directory to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Verify imports work
try:
    from scaledown import ScaleDownConfig
    print(f"✓ ScaleDown modules available (no package installation needed)")
except ImportError as e:
    print(f"✗ Error importing scaledown: {e}")
    print(f"  Make sure you're in the soft_compression/ directory")
    sys.exit(1)
