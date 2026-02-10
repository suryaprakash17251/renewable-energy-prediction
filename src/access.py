import os

# Define project paths relative to src/
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
config_path = os.path.join(base_dir, 'config', 'config.yaml')
data_path = os.path.join(base_dir, 'data', 'test_write.txt')
models_path = os.path.join(base_dir, 'models', 'trained', 'test_write.txt')
plots_path = os.path.join(base_dir, 'results', 'plots', 'test_write.txt')
metrics_path = os.path.join(base_dir, 'results', 'metrics', 'test_write.txt')

# Test config read
try:
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            f.read()
        print(f"✅ Read access confirmed for {config_path}")
    else:
        print(f"⚠ No config.yaml found at {config_path}")
except PermissionError as e:
    print(f"❌ Permission denied reading {config_path}: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Error reading {config_path}: {e}")
    exit(1)

# Test write access for each directory
test_paths = [
    (data_path, 'data'),
    (models_path, 'models/trained'),
    (plots_path, 'results/plots'),
    (metrics_path, 'results/metrics')
]

for test_path, dir_name in test_paths:
    dir_path = os.path.dirname(test_path)
    try:
        os.makedirs(dir_path, exist_ok=True)
        with open(test_path, 'w') as f:
            f.write('test')
        os.remove(test_path)
        print(f"✅ Write access confirmed for {dir_path}")
    except PermissionError as e:
        print(f"❌ Permission denied writing to {dir_path}: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Error writing to {dir_path}: {e}")
        exit(1)

print("✅ All file access tests passed successfully!")