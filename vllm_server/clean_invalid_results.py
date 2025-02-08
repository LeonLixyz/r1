import os
import json
from pathlib import Path

def is_valid_result(result):
    """Check if a result is valid by verifying parsed fields."""
    if not result.get('parsed'):
        return False
    
    parsed = result['parsed']
    # Check if all required fields exist and are non-empty
    required_fields = ['explanation', 'answer', 'confidence']
    return all(parsed.get(field) not in [None, ""] for field in required_fields)

def clean_results(results_dir='results', dry_run=True):
    """
    Check all result files and identify/delete invalid ones.
    
    Args:
        results_dir (str): Directory containing result JSON files
        dry_run (bool): If True, only print files to be deleted without actually deleting
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory {results_dir} does not exist!")
        return

    invalid_files = []
    total_files = 0

    # Check all JSON files
    for file_path in results_path.glob('*.json'):
        total_files += 1
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
            
            if not is_valid_result(result):
                invalid_files.append(file_path)
                print(f"Invalid result found: {file_path.name}")
                
        except json.JSONDecodeError:
            print(f"Corrupted JSON file found: {file_path.name}")
            invalid_files.append(file_path)
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
            invalid_files.append(file_path)

    # Report findings
    print(f"\nFound {len(invalid_files)} invalid files out of {total_files} total files")
    
    # Delete invalid files if not in dry run mode
    if not dry_run and invalid_files:
        print("\nDeleting invalid files...")
        for file_path in invalid_files:
            try:
                file_path.unlink()
                print(f"Deleted: {file_path.name}")
            except Exception as e:
                print(f"Error deleting {file_path.name}: {str(e)}")
    elif invalid_files:
        print("\nDry run mode - no files were deleted")
        print("To delete invalid files, run with --delete flag")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Clean invalid result files')
    parser.add_argument('--results-dir', type=str, default='results',
                      help='Directory containing result JSON files')
    parser.add_argument('--delete', action='store_true',
                      help='Actually delete invalid files (default is dry run)')
    args = parser.parse_args()

    clean_results(args.results_dir, dry_run=not args.delete) 