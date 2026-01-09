import os
import openai

def check_key():
    print("🔍 Checking OpenAI API Key Configuration...")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable is NOT set.")
        print("   Please set it using:")
        print("   PowerShell: $env:OPENAI_API_KEY = 'sk-proj-...'")
        print("   CMD:        set OPENAI_API_KEY=sk-proj-...")
        return

    print(f"✔ API Key found in environment variables.")
    print(f"   Key length: {len(api_key)}")
    print(f"   Key prefix: {api_key[:15]}...")
    
    if api_key.startswith('"') or api_key.startswith("'"):
         print("⚠️ WARNING: The key seems to include quotes. This might cause 401 errors.")
         print("   Please set it WITHOUT quotes in your terminal.")

    print("\nAttempting to make a simple API call...")
    
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello! Are you working?"}],
            max_tokens=10
        )
        print("\n✅ SUCCESS! The API Key is valid and working.")
        print(f"   Response: {response.choices[0].message.content}")
        
    except openai.AuthenticationError as e:
        print("\n❌ AUTHENTICATION ERROR (401):")
        print(f"   {e}")
        print("   This means OpenAI rejected the key. Check if it's revoked or has extra spaces.")
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")

if __name__ == "__main__":
    check_key()
