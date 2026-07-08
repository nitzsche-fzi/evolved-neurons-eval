import subprocess
import time

    
def get_average_smi_data(n_tries, fields=None):
    """
    Average selected numeric fields across multiple nvidia-smi snapshots.
    Averages utilization and free memory; carries static fields (name, compute_cap, total, pci) from first.
    """
    all_data = []
    for i in range(n_tries):
        data = get_nvidia_smi(fields)
        print(f"\rQuery {i+1}/{n_tries} complete.", end="", flush=True)
        if i < n_tries - 1:
            time.sleep(0.0) # wait a bit before the next query to avoid overwhelming the GPU
        if data:
            all_data.append(data)
    print()
    if not all_data:
        return {}
    first_query = all_data[0]
    avg = {}
    for gpu_idx in first_query.keys():
        name        = first_query[gpu_idx].get("name")
        compute_cap = first_query[gpu_idx].get("compute_cap")
        pci_bus_id  = first_query[gpu_idx].get("pci_bus_id")
        total_gb    = first_query[gpu_idx].get("memory_total_gb")
        util_vals = [d[gpu_idx].get("utilization_gpu") for d in all_data if gpu_idx in d]
        util_vals = [v for v in util_vals if v is not None]
        free_vals = [d[gpu_idx].get("memory_free_gb") for d in all_data if gpu_idx in d]
        free_vals = [v for v in free_vals if v is not None]
        avg[gpu_idx] = {
            "name": name,
            "compute_cap": compute_cap,
            "pci_bus_id": pci_bus_id,
            "memory_total_gb": total_gb,
            "memory_free_gb": sum(free_vals) / len(free_vals) if free_vals else None,
            "utilization_gpu": sum(util_vals) / len(util_vals) if util_vals else None,
        }
    return avg


def get_nvidia_smi(fields=None):
    """
    Query nvidia-smi for detailed per-GPU info without initializing CUDA contexts.
    Returns a dict keyed by GPU index with normalized keys:
      name, compute_cap, memory_total_gb, memory_free_gb, utilization_gpu, pci_bus_id
    You may pass a custom fields list (must include 'index' as first element).
    """
    # Fields supported by nvidia-smi
    default_fields = [
        "index",
        "name",
        "compute_cap",
        "memory.total",
        "memory.free",
        "utilization.gpu",
        "pci.bus_id",
    ]
    use_fields = fields or default_fields
    try:
        raw = _query_nvidia_smi(use_fields)
        result = {}
        for idx, vals in raw.items():
            name = vals.get("name")
            compute_cap = vals.get("compute_cap")
            total_raw = vals.get("memory.total")
            free_raw = vals.get("memory.free")
            util_raw = vals.get("utilization.gpu")
            pci_bus_id = vals.get("pci.bus_id")

            def to_gb(x):
                try:
                    v = float(x)
                    return v / 1024.0 if v > 200.0 else v
                except Exception:
                    return None

            def to_float(x):
                try:
                    return float(x)
                except Exception:
                    return None

            result[idx] = {
                "name": name,
                "compute_cap": compute_cap,
                "memory_total_gb": to_gb(total_raw),
                "memory_free_gb": to_gb(free_raw),
                "utilization_gpu": to_float(util_raw),
                "pci_bus_id": pci_bus_id,
            }
        return result
    except FileNotFoundError:
        print("nvidia-smi not found. Please ensure NVIDIA drivers are installed and in your PATH.")
        return {}
    except Exception as e:
        print(f"Error querying nvidia-smi: {e}")
        return {}

def _query_nvidia_smi(fields):
    """
    Executes nvidia-smi and parses its output to get GPU utilization.
    Returns a dictionary mapping GPU index (as per nvidia-smi) to utilization.
    """
    fmt = "--format=csv,noheader,nounits"
    query = f"nvidia-smi --query-gpu={','.join(fields)} {fmt}"
    output = subprocess.check_output(query, shell=True).decode('utf-8').strip()
    parsed = {}
    for line in output.split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != len(fields):
            continue
        idx = int(parts[0])  # first field must be index
        parsed[idx] = {fields[i]: parts[i] for i in range(len(fields))}
    return parsed


def show_gpu_dialog() -> list:
        print("Querying GPU information via nvidia-smi...")
        devices = list[int]()
        smi_avg = get_average_smi_data(20)
        if not smi_avg:
            print("No GPUs detected by nvidia-smi. Exiting.")
            exit(1)
        indices = sorted(smi_avg.keys())
        print(f"Found {len(indices)} GPUs:")
        for i in indices:
            info = smi_avg[i]
            util = info.get('utilization_gpu', -1)
            print(f"--- Device {i} ---")
            print(f"  Name: {info['name']}")
            print(f"  Capability: {info.get('compute_cap', 'unknown')}")
            print(f"  Total Memory: {info['memory_total_gb']:.2f} GB")
            print(f"  Free Memory: {info['memory_free_gb']:.2f} GB")
            print(f"  Used Memory: {(info['memory_total_gb'] - info['memory_free_gb']):.2f} GB")
            print(f"  Utilization: {util}%")
            print(f"  PCI Bus ID: {info['pci_bus_id']}")
            print("do you want to use this GPU? (y/n)")
            choice = input(f"Use GPU {i} (y/n): ").strip().lower()
            choice = choice[0] if choice else choice
            if choice == 'y':
                devices.append(i)
            elif choice == 'n':
                print(f"Skipping GPU {i}.")
            else:
                print("Invalid input, skipping this GPU.")
        if not devices:
            print("No GPUs selected. Exiting.")
            exit(1)
        print(f"Selected devices: {devices}")
        return devices