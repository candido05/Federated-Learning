好的，这里是一份完整、专业的 `README.md` 文件内容。它详细描述了您的数据集集合，并提供了使用方法。

您可以直接将以下全部内容复制到一个名为 `README.md` 的文本文件中，然后将其放入压缩包的根目录。

https://ieee-dataport.org/documents/federated-learning-dataset-driving-behavior-prediction-v2x-satellite-integrated-network
---
---

# Multi-Scale Dataset for Driving Behavior Prediction in V2X-Satellite Networks

## 1. Overview

This dataset provides a comprehensive, multi-scale simulation environment for research in Federated Learning (FL), Vehicle-to-Everything (V2X) communications, and autonomous driving behavior analysis. It contains four distinct simulation scenarios, systematically varying the vehicle density with **K=100, 200, 300, and 400 vehicles**. Each scenario is based on a 3600-second highway traffic simulation using the Intelligent Driver Model (IDM).

For each scale (K), the collection provides:
- **Private Datasets:** Per-vehicle data files formatted for federated learning experiments.
- **A Global Validation Set:** A centralized dataset for model evaluation.
- **Time-Series Snapshots:** Detailed snapshots of the entire traffic and communication (V2R and S2R) state, captured every second.

This hierarchical structure allows researchers to rigorously evaluate the scalability and performance of their models and algorithms under varying network loads and traffic densities.

## 2. Key Features

- **Multi-Scale Scenarios:** Includes four complete datasets for K=100, 200, 300, and 400 vehicles.
- **Rich Feature Set:** A 22-dimensional feature vector captures detailed ego-vehicle state and local traffic perception.
- **Federated Learning Ready:** Data is pre-partitioned by vehicle, ideal for FL simulations.
- **Comprehensive Snapshots:** Includes time-series data on vehicle kinematics, V2R (Vehicle-to-RSU), and S2R (Satellite-to-RSU) communication links.
- **Consistent Data Format:** All scenarios and data files follow the same unified structure.

## 3. File Structure

This archive contains four complete datasets, organized by vehicle count (K). The root directory after unzipping is structured as follows:

```
/
├── dataset_K100_seed42/
│   ├── simulation_config.json        # Simulation configuration for K=100
│   ├── validation_set.json           # Global validation set
│   ├── private_datasets/             # Directory for each vehicle's private dataset
│   │   ├── veh_0.json
│   │   └── ...
│   └── snapshots/                    # Directory for traffic state snapshots
│       ├── snapshot_0000.json
│       └── ...
│
├── dataset_K200_seed42/
│   └── (same internal structure as K100)
│
├── dataset_K300_seed42/
│   └── (same internal structure as K100)
│
└── dataset_K400_seed42/
    └── (same internal structure as K100)

```

## 4. Data Format Details

### 4.1. Features & Labels (`private_datasets/` and `validation_set.json`)

Each sample in these JSON files is an object with a `features` array and a `label`.

#### **Features (22-dimensional vector)**

- **Ego Vehicle State (3 features):**
  - `[0]`: Ego speed (m/s)
  - `[1]`: Ego lane index
  - `[2]`: Ego desired speed (m/s)
- **Surrounding Vehicles (5 neighbors, 3 features each = 15 features):**
  - For each neighbor (Front, Left-Front, Left-Rear, Right-Front, Right-Rear), the format is `(existence_flag, distance_m, speed_ms)`:
    - `[3, 4, 5]`: Front vehicle
    - `[6, 7, 8]`: Left-Front vehicle
    - `[9, 10, 11]`: Left-Rear vehicle
    - `[12, 13, 14]`: Right-Front vehicle
    - `[15, 16, 17]`: Right-Rear vehicle
- **Relative Speeds (4 features):**
  - `[18]`: Speed difference with front vehicle (m/s)
  - `[19]`: Speed difference with left-front vehicle (m/s)
  - `[20]`: Speed difference with right-front vehicle (m/s)
  - `[21]`: Speed difference with left-rear vehicle (m/s)

#### **Label (Discrete Driving Maneuver)**

The label represents the ego vehicle's action, categorized by its acceleration `a`:
- `0`: **Deceleration** (`a < -1.0 m/s²`)
- `1`: **Cruising** (`-1.0 ≤ a ≤ 1.0 m/s²`)
- `2`: **Acceleration** (`a > 1.0 m/s²`)

### 4.2. Traffic State Snapshots (`snapshots/`)

Each `snapshot_XXXX.json` file captures the state of the entire simulation at a specific timestamp.
- `timestamp_s`: The simulation time in seconds.
- `vehicle_states`: A list of objects, each containing a vehicle's `id`, `position_m`, `speed_ms`, `lane`, and `desired_speed_ms`.
- `rsu_coverage`: A dictionary where keys are RSU IDs. Each value is a list of vehicles within that RSU's coverage, including `vehicle_id`, `distance_m`, and `path_loss_dB`.
- `satellite_visibility`: A dictionary where keys are RSU IDs. Each value is a list of visible satellites, including `sat_id`, `elevation_deg`, `distance_km`, and `path_loss_dB`.

### 4.3. Simulation Configuration (`simulation_config.json`)

This file contains all the parameters used to generate the corresponding dataset, including mobility models (IDM), channel models (V2R, S2R), and general simulation settings.

## 5. Example Usage (Python)

The data is stored in JSON format and can be easily loaded in any modern programming language.

```python
import json
import os

# --- Example 1: Load a private dataset for one vehicle ---
k_value = 100
vehicle_id = 0

private_data_path = os.path.join(
    f'dataset_K{k_value}_seed42', 
    'private_datasets', 
    f'veh_{vehicle_id}.json'
)

with open(private_data_path, 'r') as f:
    private_data = json.load(f)

# Access the first data point
first_sample = private_data[0]
features = first_sample['features']
label = first_sample['label']

print(f"--- K={k_value}, Vehicle {vehicle_id} ---")
print(f"Number of private samples: {len(private_data)}")
print(f"Features of first sample: {features}")
print(f"Label of first sample: {label}\n")


# --- Example 2: Load the global validation set ---
validation_path = os.path.join(
    f'dataset_K{k_value}_seed42',
    'validation_set.json'
)

with open(validation_path, 'r') as f:
    validation_data = json.load(f)

print(f"--- K={k_value}, Global Validation Set ---")
print(f"Number of validation samples: {len(validation_data)}")
print(f"Features of first validation sample: {validation_data[0]['features']}")
```

## 6. Potential Applications

This dataset is suitable for, but not limited to, the following research areas:
- **Scalability Analysis:** Evaluating how FL algorithms, communication protocols, or driving models perform as vehicle density increases.
- **Federated Learning:** Training and testing FL algorithms for tasks like driving maneuver prediction or anomaly detection.
- **Centralized Machine Learning:** Developing robust, generalizable models for driving behavior prediction.
- **V2X & Satellite Communication Analysis:** Studying the dynamics of communication links in mobile networks of varying scales.
- **Traffic Flow Simulation and Analysis:** Investigating macroscopic and microscopic traffic phenomena.

## 7. License

This dataset is released under the [Your Chosen License, e.g., Creative Commons Attribution 4.0 International (CC BY 4.0)] license.

## 8. Citation

If you use this dataset in your research, please cite it as follows:

```
[Author(s). (Year). Title of Dataset (Version). Publisher/Repository. DOI:xxxxxxxx]
(Note: A specific DOI will be assigned by IEEE DataPort upon publication. Please update this section accordingly.)
```