import yaml
import os
import sys

def clean_yaml(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    backup_path = file_path + '.bak'
    
    # Always create a fresh backup of the current state before processing
    import shutil
    shutil.copy2(file_path, backup_path)
    print(f"Backup updated at {backup_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        print("Error: YAML root must be a list of elements.")
        return

    original_count = len(data)
    # Filter out elements where 'text' is just a single dot
    filtered_data = [item for item in data if item.get('text') != '.']
    removed_count = original_count - len(filtered_data)

    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(filtered_data, f, allow_unicode=True, sort_keys=False)

    print(f"Successfully removed {removed_count} elements from {file_path}. File updated.")

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else 'ruta_final_original_lang.yaml'
    clean_yaml(target_file)
