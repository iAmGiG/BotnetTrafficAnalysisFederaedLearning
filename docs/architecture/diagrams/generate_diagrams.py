"""
Master Architecture Diagram Generator

This script generates both CURRENT and TARGET system architecture diagrams.

Usage:
    python generate_diagrams.py              # Generate both
    python generate_diagrams.py --current    # Current only
    python generate_diagrams.py --target     # Target only
"""

import sys
from pathlib import Path

# Import diagram generators
from generate_current_diagrams import main as generate_current
from generate_target_diagrams import main as generate_target


def print_header(text):
    """Print a section header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def main():
    """Generate architecture diagrams based on command line args."""
    args = sys.argv[1:] if len(sys.argv) > 1 else []

    # Determine what to generate
    gen_current = '--current' in args or not args
    gen_target = '--target' in args or not args

    print_header("Architecture Diagram Generator")
    print("This will generate diagrams for:")

    if gen_current:
        print("  [*] CURRENT system (2020-era, as-is)")
    if gen_target:
        print("  [*] TARGET system (2025 modernized, to-be)")

    print()

    # Generate current system diagrams
    if gen_current:
        print_header("CURRENT System Diagrams (As-Is)")
        try:
            generate_current()
            print("\n[SUCCESS] Current system diagrams completed successfully!")
        except Exception as e:
            print(f"\n[ERROR] Error generating current diagrams: {e}")
            import traceback
            traceback.print_exc()

    # Generate target system diagrams
    if gen_target:
        print_header("TARGET System Diagrams (To-Be)")
        try:
            generate_target()
            print("\n[SUCCESS] Target system diagrams completed successfully!")
        except Exception as e:
            print(f"\n[ERROR] Error generating target diagrams: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print_header("Summary")
    output_dir = Path(__file__).parent.parent / 'images'

    print(f"Diagrams saved to: {output_dir}/")
    print()

    if gen_current:
        print(f"Current system (2020): {output_dir}/current/")
        print("  - current_system_architecture.png")
        print("  - current_file_structure.png")
        print("  - current_data_pipeline.png")
        print("  - current_dependencies.png")
        print()

    if gen_target:
        print(f"Target system (2025): {output_dir}/")
        print("  - target_system_architecture.png")
        print("  - target_data_pipeline.png")
        print("  - target_autoencoder_architecture.png")
        print("  - target_classifier_architecture.png")
        print("  - target_federated_architecture.png")
        print("  - target_deployment_architecture.png")
        print("  - target_file_structure.png")
        print()

    print("Note: .gv source files are local only (not committed to git)")
    print()

    print("Use these diagrams in documentation:")
    print("  - README.md")
    print("  - docs/architecture/ARCHITECTURE.md")
    print("  - GitHub issues")
    print("  - Presentations")
    print()
    print("[SUCCESS] All done!")


if __name__ == '__main__':
    main()
