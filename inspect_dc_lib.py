
import datacommons as dc
import inspect

def inspect_dc():
    print("Inspecting 'datacommons' module...")
    print(f"Version: {getattr(dc, '__version__', 'unknown')}")
    
    # List all functions in top level
    functions = [o[0] for o in inspect.getmembers(dc, inspect.isfunction)]
    print("\nFunctions available:")
    for f in functions:
        print(f" - {f}")

if __name__ == "__main__":
    inspect_dc()
