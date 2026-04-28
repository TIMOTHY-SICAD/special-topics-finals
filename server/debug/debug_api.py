import requests
import json

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# Check main category structure
params = {
    'action': 'query',
    'list': 'categorymembers',
    'cmtitle': 'Category:Road signs in the Philippines',
    'cmlimit': '500',
    'format': 'json'
}

response = requests.get('https://commons.wikimedia.org/w/api.php', params=params, headers=headers)
data = response.json()
members = data['query']['categorymembers']

# Categorize by namespace
files = [m for m in members if m.get('ns') == 6]  # File namespace
categories = [m for m in members if m.get('ns') == 14]  # Category namespace

print(f"Main category 'Road signs in the Philippines':")
print(f"  Direct files: {len(files)}")
print(f"  Subcategories: {len(categories)}")

if files:
    print(f"\n  First few files:")
    for f in files[:5]:
        print(f"    - {f.get('title')}")

print(f"\n  Subcategories to explore:")
for c in categories[:10]:
    print(f"    - {c.get('title')}")

# Now recursively check all subcategories for files
print("\n" + "="*60)
print("Scanning subcategories for files...")
print("="*60)

all_files = files.copy()
for category in categories[:10]:  # Check first 10 subcategories
    cat_title = category.get('title')
    sub_params = {
        'action': 'query',
        'list': 'categorymembers',
        'cmtitle': cat_title,
        'cmtype': 'file',
        'cmlimit': '100',
        'format': 'json'
    }
    
    sub_response = requests.get('https://commons.wikimedia.org/w/api.php', params=sub_params, headers=headers)
    sub_data = sub_response.json()
    sub_files = sub_data['query']['categorymembers']
    
    if sub_files:
        print(f"\n{cat_title}: {len(sub_files)} files")
        for f in sub_files[:3]:
            print(f"  - {f.get('title')}")
        all_files.extend(sub_files)

print(f"\n\nTotal unique files found so far: {len(set(f.get('title') for f in all_files))}")

# Sample a file to check if we can get thumbnail
if all_files:
    sample = all_files[0]
    file_title = sample.get('title')
    print(f"\n\nTesting thumbnail API for: {file_title}")
    
    thumb_params = {
        'action': 'query',
        'titles': file_title,
        'prop': 'imageinfo',
        'iiprop': 'url|thumburl',
        'iiurlwidth': '400',
        'format': 'json'
    }
    
    thumb_response = requests.get('https://commons.wikimedia.org/w/api.php', params=thumb_params, headers=headers)
    thumb_data = thumb_response.json()
    
    for page in thumb_data['query']['pages'].values():
        if 'imageinfo' in page:
            info = page['imageinfo'][0]
            print(f"  URL: {info.get('url', 'N/A')[:80]}")
            print(f"  Thumb URL: {info.get('thumburl', 'N/A')[:80]}")
