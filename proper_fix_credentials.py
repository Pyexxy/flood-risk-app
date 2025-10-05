import json

# Read the current credentials
with open('gee-credentials.json', 'r') as f:
    content = f.read()
    creds = json.loads(content)

# Check current state of private key
private_key = creds['private_key']

print("Current private_key format:")
print(f"First 60 chars: {private_key[:60]}")
print(f"Has \\\\n (double-escaped): {'\\\\n' in repr(private_key)}")
print(f"Has \\n (single-escaped): {'\\n' in private_key and '\\\\n' not in repr(private_key)}")

# If it has double-escaped newlines (\\n in the string representation), fix it
if '\\n' in private_key:
    print("\n✓ Private key format is correct (has literal \\n characters)")
    print("No changes needed.")
else:
    print("\n✗ Private key has actual newlines or double-escapes")
    print("This shouldn't happen with properly formatted JSON")

    # Try to fix by ensuring proper JSON string format
    # The private key should have \n as escape sequences in the JSON string
    if '\n' in private_key:
        # Has actual newlines, need to escape them
        print("Converting actual newlines to \\n escape sequences...")
        # This is wrong - the JSON should already have them escaped
        print("ERROR: The JSON file is malformed. Please re-download from Google Cloud.")