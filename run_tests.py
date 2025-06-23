"""
Test runner script for PheTK.
Provides convenient commands to run different types of tests.
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle the output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"Command failed with return code: {result.returncode}")
            return False
        else:
            print("Command completed successfully!")
            return True
            
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def main():
    """Main test runner function."""
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py [test_type]")
        print("Available test types:")
        print("  all        - Run all tests")
        print("  unit       - Run unit tests only")
        print("  integration - Run integration tests only")
        print("  coverage   - Run tests with coverage report")
        print("  specific   - Run specific test file (requires filename)")
        print("  lint       - Run code quality checks")
        print("  fast       - Run fast tests only (exclude slow tests)")
        return
    
    test_type = sys.argv[1].lower()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    success = True
    
    if test_type == "all":
        print("Running all tests...")
        success &= run_command(
            "python -m pytest tests/ -v --tb=short",
            "All Tests"
        )
        
    elif test_type == "unit":
        print("Running unit tests...")
        success &= run_command(
            "python -m pytest tests/ -v -m 'unit or not integration' --tb=short",
            "Unit Tests"
        )
        
    elif test_type == "integration":
        print("Running integration tests...")
        success &= run_command(
            "python -m pytest tests/test_integration.py -v --tb=short",
            "Integration Tests"
        )
        
    elif test_type == "coverage":
        print("Running tests with coverage...")
        success &= run_command(
            "python -m pytest tests/ --cov=src/PheTK --cov-report=html --cov-report=term-missing --cov-report=xml -v",
            "Coverage Tests"
        )
        if success:
            print("\nCoverage report generated in htmlcov/index.html")
            
    elif test_type == "specific":
        if len(sys.argv) < 3:
            print("Please specify a test file name")
            print("Example: python run_tests.py specific test_utils.py")
            return
        
        test_file = sys.argv[2]
        if not test_file.startswith("test_"):
            test_file = f"test_{test_file}"
        if not test_file.endswith(".py"):
            test_file = f"{test_file}.py"
            
        print(f"Running specific test file: {test_file}")
        success &= run_command(
            f"python -m pytest tests/{test_file} -v --tb=short",
            f"Specific Test: {test_file}"
        )
        
    elif test_type == "lint":
        print("Running code quality checks...")
        
        # Check if flake8 is available
        try:
            import flake8
            success &= run_command(
                "python -m flake8 src/PheTK/ --max-line-length=88 --extend-ignore=E203,W503",
                "Flake8 Linting"
            )
        except ImportError:
            print("Flake8 not available, skipping linting")
        
        # Check if black is available
        try:
            import black
            success &= run_command(
                "python -m black --check --diff src/PheTK/",
                "Black Code Formatting Check"
            )
        except ImportError:
            print("Black not available, skipping formatting check")
            
    elif test_type == "fast":
        print("Running fast tests only...")
        success &= run_command(
            "python -m pytest tests/ -v -m 'not slow' --tb=short",
            "Fast Tests"
        )
        
    else:
        print(f"Unknown test type: {test_type}")
        print("Use 'python run_tests.py' without arguments to see available options")
        return
    
    if success:
        print(f"\n✅ {test_type.capitalize()} tests completed successfully!")
    else:
        print(f"\n❌ {test_type.capitalize()} tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
