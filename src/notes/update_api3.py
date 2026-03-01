# Update app.py with proper formatting

with open(r'E:\StockandCrypto\src\notes\app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and update the crypto_predictions function
new_lines = []
i = 0
modified = False

while i < len(lines):
    line = lines[i]
    
    # Find "normalized = request.args.get" in crypto_predictions context
    if 'normalized = request.args.get("normalized", "true").lower() != "false"' in line and not modified:
        # Check if next lines are "data_path = _get_data_path()"
        if i + 2 < len(lines) and 'data_path = _get_data_path()' in lines[i + 2]:
            new_lines.append(line)  # normalized line
            new_lines.append('\n')  # blank line
            new_lines.append(' # Check cache first\n')
            new_lines.append(' cache_key = _cache_key("crypto_pred", symbol_filter or "all", limit, normalized)\n')
            new_lines.append(' cached, found = _cache_get(cache_key)\n')
            new_lines.append(' if found:\n')
            new_lines.append('     return jsonify(cached)\n')
            new_lines.append('\n')
            # Skip the next blank line and data_path line, add them after
            i += 1  # skip blank line
            new_lines.append(lines[i])  # blank line
            i += 1  # skip to data_path
            new_lines.append(lines[i])  # data_path line
            modified = True
            print(f"Added cache check at line {i}")
        else:
            new_lines.append(line)
    elif 'return jsonify({' in line and 'session_forecast_blocks' in lines[i + 5] if i + 5 < len(lines) else False:
        # This is the return statement in crypto_predictions
        # Replace with cache_set + return
        # Keep the jsonify structure but add cache_set before it
        # Collect the full jsonify block
        jsonify_lines = [line]
        j = i + 1
        while j < len(lines) and '})' not in lines[j]:
            jsonify_lines.append(lines[j])
            j += 1
        if j < len(lines):
            jsonify_lines.append(lines[j])  # the }) line
        
        # Replace with result + cache_set + return
        new_lines.append(' result = {\n')
        for jl in jsonify_lines[1:-1]:  # skip the opening return jsonify({ and closing })
            new_lines.append(jl)
        new_lines.append(' }\n')
        new_lines.append(' _cache_set(cache_key, result)\n')
        new_lines.append(' return jsonify(result)\n')
        
        i = j  # skip to after })
        print(f"Updated return statement at line {i}")
    else:
        new_lines.append(line)
    
    i += 1

# Write back
with open(r'E:\StockandCrypto\src\notes\app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("File updated successfully")
