"""Read-only root check of the saved editorial candidate; never regenerate it."""
import json
from build_and_check import check

if __name__ == '__main__':
    print(json.dumps(check(),ensure_ascii=False,indent=2))
