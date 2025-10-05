import json

# Read the current credentials
with open('gee-credentials.json', 'r') as f:
    creds = json.load(f)

# The private key should have literal \n characters, not actual newlines
# This ensures proper formatting
private_key = creds['private_key']

# If it doesn't have \n, it's already broken into lines - let's fix it
if '\\n' not in private_key and '\n' in private_key:
    # Replace actual newlines with \n string
    private_key = private_key.replace('\n', '\\n')
    creds['private_key'] = private_key

    # Save the fixed version
    with open('gee-credentials-fixed.json', 'w') as f:
        json.dump(creds, f, indent=2)

    print("✓ Fixed credentials saved to gee-credentials-fixed.json")
    print("Replace your original file with this one")
else:
    print("Credentials format looks correct")
    print(f"Private key starts with: {private_key[:50]}")