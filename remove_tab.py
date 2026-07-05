import sys

with open('app.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    # Remove tab from st.tabs assignment
    if line.strip() == '"Scenario Analysis",':
        continue
    
    # Remove tab_scenarios from tuple
    if 'tab_guide, tab_validation, tab_scenarios, tab_performance, tab_vppa_8760 =' in line:
        line = line.replace('tab_scenarios, ', '')
    
    if line.startswith('with tab_scenarios:'):
        skip = True
        continue
        
    if skip:
        # if line is empty or starts with space/tab, it's inside the block
        if line.strip() == '' or line.startswith(' ') or line.startswith('\t'):
            continue
        else:
            skip = False
            
    if not skip:
        new_lines.append(line)

with open('app.py', 'w') as f:
    f.writelines(new_lines)

print("Done removing Scenario Analysis tab code.")
