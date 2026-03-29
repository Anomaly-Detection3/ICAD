'''
#The monitoring process for a single GPU#
We are using a single GPU 3080Ti
'''
import threading
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] =42

try:
    import torch
except ImportError:
    print("PyTorch is not installed. This function requires PyTorch to monitor GPU memory.")
    torch = None


def monitor_gpu_memory(duration_minutes=60, interval_seconds=1, save_path="gpu_memory_plots", experiment_name=None,
                       stop_event=None):
    """
    Parameters:
    duration_minutes (float): Max monitoring duration in minutes.
    interval_seconds (int): Sampling interval in seconds.
    save_path (str): Directory to save charts.
    experiment_name (str): Experiment name for title and filename.
    stop_event (threading.Event): An event to signal the thread to stop.
    """
    if not torch or not torch.cuda.is_available():
        print("PyTorch with CUDA is not available. Aborting GPU monitoring.")
        return {}

    os.makedirs(save_path, exist_ok=True)

    if experiment_name is None:
        experiment_name = f"GPU_Memory_Monitor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    start_time = time.time()
    stop_time = start_time + (duration_minutes * 60)

    timestamps = []
    memory_used = []

    print(f"[{experiment_name}] Starting GPU memory monitoring...")
    print(f"Max duration: {duration_minutes:.1f} minutes. Will stop when main task finishes.")

    try:
        while time.time() < stop_time:

            if stop_event and stop_event.is_set():
                print(f"[{experiment_name}] Stop signal received. Finishing monitoring.")
                break

            current_time = time.time() - start_time
            memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)  # MB

            timestamps.append(current_time / 60)
            memory_used.append(memory_allocated)

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print(f"\n[{experiment_name}] Monitoring interrupted by user. Generating plot...")

    if not timestamps or not memory_used:
        print("No data collected. Exiting without generating a plot.")
        return {}


    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    peak_memory_mb = max(memory_used) if memory_used else 0
    avg_memory_mb = np.mean(memory_used) if memory_used else 0
    utilization = (peak_memory_mb / total_gpu_memory) * 100 if total_gpu_memory > 0 else 0
    peak_time_min = timestamps[np.argmax(memory_used)] if memory_used else 0


    use_gb = peak_memory_mb > 1536
    if use_gb:
        unit, divisor = "GB", 1024
        memory_used = [mem / divisor for mem in memory_used]
        peak_memory, avg_memory, total_memory = peak_memory_mb / divisor, avg_memory_mb / divisor, total_gpu_memory / divisor
    else:
        unit, divisor = "MB", 1
        peak_memory, avg_memory, total_memory = peak_memory_mb, avg_memory_mb, total_gpu_memory


    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 9), dpi=500, facecolor='#F5F5F5')


    colors = {
        'main': '#00ffff',
        'fill': '#e4cdff',
        'peak': '#FF6B6B',
        'avg': '#6BCB77',
        'highlight': '#FF9E3F',
        'text_bg': 'white'
    }


    ax.plot(timestamps, memory_used,
            color=colors['main'],
            linewidth=3.5,
            label='Allocated Memory',
            alpha=0.9,
            zorder=3)

    ax.fill_between(timestamps, memory_used,
                    color=colors['fill'],
                    alpha=0.25,
                    zorder=2)


    ax.axhline(y=peak_memory,
               color=colors['peak'],
               linestyle='--',
               linewidth=2.5,
               alpha=0.8,
               label=f'Peak: {peak_memory:.2f} {unit}',
               zorder=1)

    ax.axhline(y=avg_memory,
               color=colors['avg'],
               linestyle='-.',
               linewidth=2.0,
               alpha=0.8,
               label=f'Avg: {avg_memory:.2f} {unit}',
               zorder=1)


    # ax.plot(peak_time_min, peak_memory,
    #         'o',
    #         color=colors['highlight'],
    #         markersize=12,
    #         markeredgecolor='white',
    #         markeredgewidth=2,
    #         label=f'Peak Time: {peak_time_min:.1f} min',
    #         zorder=4)


    stats_text = (f"Peak Memory: {peak_memory:.2f} {unit}\n"
                  f"Average Memory: {avg_memory:.2f} {unit}\n"
                  f"Total Device Memory: {total_memory:.2f} {unit}\n"
                  f"Peak Utilization: {utilization:.2f}%")

    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=18,
            fontfamily='DejaVu Sans',
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5',
                      fc=colors['text_bg'],
                      ec='#DDDDDD',
                      alpha=0.9,
                      lw=1.5),
            zorder=5)


    ax.set_xlabel('Time (minutes)',
                  fontsize=18,
                  fontweight='bold',
                  labelpad=12,
                  color='#333333')

    ax.set_ylabel(f'Memory Usage ({unit})',
                  fontsize=18,
                  fontweight='bold',
                  labelpad=12,
                  color='#333333')


    ax.set_ylim(bottom=0, top=max(peak_memory * 1.2, total_memory * 0.1, 1))
    ax.set_xlim(left=0, right=max(timestamps) if timestamps else 1)


    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.tick_params(axis='both',
                   which='major',
                   labelsize=16,
                   colors='#555555')


    ax.grid(True, linestyle='--', alpha=0.6, color='#E0E0E0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.5)


    ax.legend(loc='upper left',
              fontsize=18,
              framealpha=0.95,
              edgecolor='#DDDDDD',
              facecolor='white')


    fig.tight_layout(pad=2.5)
    fig.patch.set_facecolor('#F5F5F5')


    filename = experiment_name.replace(' ', '_').replace('/', '-')
    png_path = os.path.join(save_path, f"{filename}_memory_usage1.png")
    plt.savefig(png_path, dpi=500, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\nChart saved successfully to: {png_path}")
    plt.close(fig)

    return {
        "timestamps_min": timestamps,
        "memory_used": memory_used,
        "unit": unit,
        "peak_memory": peak_memory,
        "average_memory": avg_memory,
        "memory_utilization_percent": utilization
    }



'''

#The monitoring process for multiple GPUs#

'''

import torch
import threading
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import os

def monitor_gpu_memory(duration_minutes=10, save_path="gpu_memory_plots"):
    os.makedirs(save_path, exist_ok=True)

    start_time = time.time()
    stop_time = start_time + (duration_minutes * 60)
    memory_data = []

    def log_memory():
        if time.time() < stop_time:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()

                for i in range(device_count):
                    allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                    reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                    max_allocated = torch.cuda.max_memory_allocated(i) / (1024 * 1024)

                    elapsed = time.time() - start_time
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    print(
                        f"[GPU:{i} at {timestamp}] Runtime: {elapsed:.1f}s | Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Peak: {max_allocated:.2f} MB")

                    memory_data.append({
                        'time': elapsed,
                        'gpu': i,
                        'allocated_mb': allocated,
                        'reserved_mb': reserved,
                        'max_allocated_mb': max_allocated,
                        'timestamp': timestamp
                    })
            else:
                print("CUDA not available, GPU monitoring disabled")
                return

            threading.Timer(30.0, log_memory).start()
        else:
            print("\n===== GPU Memory Report =====")
            if memory_data and torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    gpu_data = [item for item in memory_data if item['gpu'] == i]
                    if gpu_data:
                        peak_allocated = max(item['allocated_mb'] for item in gpu_data)
                        peak_reserved = max(item['reserved_mb'] for item in gpu_data)
                        print(f"GPU:{i} Peak allocated: {peak_allocated:.2f} MB")
                        print(f"GPU:{i} Peak reserved: {peak_reserved:.2f} MB")
                        print(f"GPU:{i} Max memory used: {max(item['max_allocated_mb'] for item in gpu_data):.2f} MB")

                generate_plots(memory_data, device_count, save_path)
            print("============================\n")

    def generate_plots(data, device_count, save_path):
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        for i in range(device_count):
            gpu_data = [item for item in data if item['gpu'] == i]
            if not gpu_data:
                continue

            times = [item['time'] for item in gpu_data]
            allocated = [item['allocated_mb'] for item in gpu_data]
            reserved = [item['reserved_mb'] for item in gpu_data]
            max_allocated = [item['max_allocated_mb'] for item in gpu_data]

            # Plot 1: Memory Timeline
            plt.figure(figsize=(12, 6))
            plt.plot(times, allocated, 'b-', label='Allocated')
            plt.plot(times, reserved, 'r-', label='Reserved')
            plt.plot(times, max_allocated, 'g--', label='Peak')

            plt.title(f'GPU:{i} Memory Usage')
            plt.xlabel('Runtime (seconds)')
            plt.ylabel('Memory (MB)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            max_point_idx = allocated.index(max(allocated))
            plt.annotate(f'{max(allocated):.2f}MB',
                        xy=(times[max_point_idx], allocated[max_point_idx]),
                        xytext=(times[max_point_idx] + 10, allocated[max_point_idx] + 10),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

            plt.tight_layout()
            plt.savefig(f'{save_path}/gpu{i}_memory_usage_{date_str}.png', dpi=300)
            plt.savefig(f'{save_path}/gpu{i}_memory_usage_{date_str}.pdf')
            plt.close()

            # Plot 2: Pie Chart
            plt.figure(figsize=(8, 8))
            avg_allocated = np.mean(allocated)
            max_used = max(allocated)
            gpu_info = torch.cuda.get_device_properties(i)
            total_memory = gpu_info.total_memory / (1024 * 1024)  # Fixed closing parenthesis

            labels = ['Average Usage', 'Peak Usage', 'Available Memory']
            sizes = [avg_allocated, max_used - avg_allocated, total_memory - max_used]
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            explode = (0.1, 0, 0)

            plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=90)
            plt.axis('equal')
            plt.title(f'GPU:{i} Memory Distribution\n(Total: {total_memory:.2f}MB)')

            plt.tight_layout()
            plt.savefig(f'{save_path}/gpu{i}_memory_distribution_{date_str}.png', dpi=300)
            plt.savefig(f'{save_path}/gpu{i}_memory_distribution_{date_str}.pdf')
            plt.close()

            # Plot 3: Academic Style
            plt.figure(figsize=(9, 5))
            plt.plot(times, allocated, 'b-', linewidth=2, label='Live Memory')
            plt.fill_between(times, allocated, alpha=0.2, color='blue')

            plt.axhline(y=avg_allocated, color='r', linestyle='--',
                        label=f'Average: {avg_allocated:.2f}MB')

            plt.axhline(y=max_used, color='g', linestyle='-.',
                        label=f'Peak: {max_used:.2f}MB')

            plt.axhline(y=total_memory, color='k', linestyle=':',
                        label=f'Total: {total_memory:.2f}MB')

            plt.xlabel('Runtime (seconds)', fontsize=12)
            plt.ylabel('Memory Usage (MB)', fontsize=12)
            plt.title(f'Model Memory Usage on GPU:{i}', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    fancybox=True, shadow=True, ncol=4)

            memory_usage_percent = (max_used / total_memory) * 100
            plt.annotate(f'Utilization: {memory_usage_percent:.2f}%',
                        xy=(0.02, 0.02), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

            plt.tight_layout()
            plt.savefig(f'{save_path}/gpu{i}_memory_academic_{date_str}.png', dpi=300)
            plt.savefig(f'{save_path}/gpu{i}_memory_academic_{date_str}.pdf')
            plt.close()

        # Multi-GPU Comparison
        if device_count > 1:
            plt.figure(figsize=(12, 6))

            for i in range(device_count):
                gpu_data = [item for item in data if item['gpu'] == i]
                if not gpu_data:
                    continue

                times = [item['time'] for item in gpu_data]
                allocated = [item['allocated_mb'] for item in gpu_data]

                plt.plot(times, allocated, marker='o', markersize=3,
                        label=f'GPU:{i} Usage')

            plt.title('Multi-GPU Memory Comparison')
            plt.xlabel('Runtime (seconds)')
            plt.ylabel('Memory Usage (MB)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            plt.tight_layout()
            plt.savefig(f'{save_path}/multi_gpu_comparison_{date_str}.png', dpi=300)
            plt.savefig(f'{save_path}/multi_gpu_comparison_{date_str}.pdf')
            plt.close()

        print(f"\nPlots saved to: {save_path}")

    print(f"Starting GPU memory monitoring for {duration_minutes} minutes...")
    log_memory()
    return memory_data



