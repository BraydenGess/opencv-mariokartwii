import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_valid_psm_modes():
    """Return list of PSM modes that are most relevant for text line detection"""
    return [1, 3, 4, 6, 7, 9, 10, 11, 12]

def analyze_psm_reports(metrics_dir):
    # Get all PSM report files
    report_files = list(Path(metrics_dir).glob("psm_detection_report_*.json"))
    
    # Initialize data structure for aggregating results
    psm_stats = {i: {
        'total_detection_rate': [],
        'true_positive_rate': [],
        'false_positive_rate': []
    } for i in get_valid_psm_modes()}
    
    # Process each report file
    for report_file in report_files:
        with open(report_file) as f:
            data = json.load(f)
            
        # Process each PSM mode
        for psm, results in data.items():
            psm = int(psm)
            if psm not in get_valid_psm_modes():
                continue
                
            # Get total detection rate
            total_detection_rate = results['course_detection_rate']
            
            # Find the most detected course (assumed to be true positive)
            course_detections = results.get('course_detections', {})
            if course_detections:
                most_detected = max(course_detections.items(), 
                                  key=lambda x: x[1]['count'])
                true_positive_rate = most_detected[1]['ratio']
                false_positive_rate = total_detection_rate - true_positive_rate
            else:
                true_positive_rate = 0
                false_positive_rate = 0
            
            # Store the rates
            psm_stats[psm]['total_detection_rate'].append(total_detection_rate)
            psm_stats[psm]['true_positive_rate'].append(true_positive_rate)
            psm_stats[psm]['false_positive_rate'].append(false_positive_rate)
    
    # Calculate averages and create DataFrame
    df_data = []
    for psm in get_valid_psm_modes():
        stats = psm_stats[psm]
        avg_true_rate = sum(stats['true_positive_rate']) / len(stats['true_positive_rate'])
        avg_false_rate = sum(stats['false_positive_rate']) / len(stats['false_positive_rate'])
        
        # Calculate FP:TP ratio (handle division by zero)
        fp_tp_ratio = avg_false_rate / avg_true_rate if avg_true_rate > 0 else float('inf')
        
        df_data.append({
            'PSM': psm,
            'Total Detection Rate': sum(stats['total_detection_rate']) / len(stats['total_detection_rate']),
            'True Positive Rate': avg_true_rate,
            'False Positive Rate': avg_false_rate,
            'FP:TP Ratio': fp_tp_ratio
        })
    
    df = pd.DataFrame(df_data)
    
    # Create visualization with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[2, 1])
    
    # Plot 1: Detection rates
    x = np.arange(len(get_valid_psm_modes()))
    bar_width = 0.25
    
    bars1 = ax1.bar(x - bar_width, df['True Positive Rate'], 
                    bar_width, label='True Positive Rate', color='green', alpha=0.7)
    bars2 = ax1.bar(x, df['False Positive Rate'],
                    bar_width, label='False Positive Rate', color='red', alpha=0.7)
    bars3 = ax1.bar(x + bar_width, df['Total Detection Rate'],
                    bar_width, label='Total Detection Rate', color='blue', alpha=0.7)
    
    # Add value labels on top of bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom', rotation=90)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # Customize first plot
    ax1.set_xlabel('PSM Mode')
    ax1.set_ylabel('Rate')
    ax1.set_title('Detection Rates by PSM Mode\nAveraged Across All Courses')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(get_valid_psm_modes())
    
    # Plot 2: FP:TP Ratio
    bars4 = ax2.bar(x, df['FP:TP Ratio'], color='purple', alpha=0.7)
    
    # Add value labels for ratio bars
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Customize second plot
    ax2.set_xlabel('PSM Mode')
    ax2.set_ylabel('False Positive : True Positive Ratio')
    ax2.set_title('False Positive to True Positive Ratio by PSM Mode')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(get_valid_psm_modes())
    
    # Add vertical separators between PSM groups in both plots
    for i in range(len(get_valid_psm_modes())-1):
        ax1.axvline(x=i+0.5, color='gray', linestyle='--', alpha=0.3)
        ax2.axvline(x=i+0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Print summary statistics
    print("\nPSM Mode Analysis Summary:")
    print("=" * 50)
    for _, row in df.iterrows():
        print(f"\nPSM {int(row['PSM'])}:")
        print(f"True Positive Rate:  {row['True Positive Rate']:.2%}")
        print(f"False Positive Rate: {row['False Positive Rate']:.2%}")
        print(f"Total Detection Rate: {row['Total Detection Rate']:.2%}")
        print(f"FP:TP Ratio: {row['FP:TP Ratio']:.2f}")
    
    # Save plot
    plt.tight_layout()
    plt.savefig('psm_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

if __name__ == "__main__":
    metrics_dir = "metrics"  # Change this if your metrics are stored elsewhere
    df = analyze_psm_reports(metrics_dir)