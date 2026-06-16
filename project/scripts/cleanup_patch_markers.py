import os
import shutil

ROOT = r"d:/workspace/project"
PATTERNS = ("***", "@@")

# Also allow markers that appear with surrounding spaces
PATTERNS_EXTENDED = tuple(p for p in PATTERNS)

def clean_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = [ln for ln in lines if not any(ln.strip().startswith(p) for p in PATTERNS)]
    if len(new_lines) != len(lines):
        shutil.copy2(path, path + '.bak')
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return True
    return False

def main():
    changed = []
    for dirpath, dirnames, files in os.walk(ROOT):
        for fn in files:
            if fn.endswith('.py'):
                fp = os.path.join(dirpath, fn)
                try:
                    if clean_file(fp):
                        changed.append(os.path.relpath(fp, ROOT).replace('\\\\','/'))
                except Exception as e:
                    print('ERR', fp, e)
    if changed:
        print('CLEANED')
        for p in changed:
            print(p)
    else:
        print('NO_CHANGES')

if __name__ == '__main__':
    main()
