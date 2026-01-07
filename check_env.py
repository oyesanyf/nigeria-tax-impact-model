
import os

def check_env():
    print("Checking environment variable 'DC_API_KEY'...")
    key = os.environ.get("DC_API_KEY")
    
    if key:
        print(f"✅ SUCCESS: DC_API_KEY is set!")
        print(f"   Length: {len(key)} characters")
        print(f"   Starts with: {key[:5]}...")
    else:
        print("❌ FAILURE: DC_API_KEY is NOT set in this shell session.")
        print("   Please run: $env:DC_API_KEY='your_key' (PowerShell)")
        print("   or: set DC_API_KEY=your_key (CMD)")

if __name__ == "__main__":
    check_env()
