import numpy as np
import pandas as pd  # For reading CSV
import os  # For joining path components


SENSOR_METADATA = {
    "AIT": "Analyzer Indicator Transmitter",
    "FIT": "Flow Indicator Transmitter",
    "LIT": "Level Indicator Transmitter",
    "MV": "Motorized Valve",
    "P": "Pump",
    "UV": "UV System",
    "PIT": "Pressure Indicator Transmitter",
    "DPIT": "Differential Pressure Indicator Transmitter"
}


TREND_DIFF_THRESHOLD_ABS = 0.5


VOLATILITY_RANGE_THRESHOLDS = {'low': 1.0, 'medium': 5.0}


BINARY_CHANGE_THRESHOLD = 0
BINARY_TYPES = ['P', 'UV', 'MV']


TOP_N_SENSORS = 3


GROUP_BEHAVIOR_THRESHOLD_RATIO = 0.5
MAX_GROUP_SUMMARIES_DISPLAYED = 2


WEIGHT_TREND_SCORE = 1.0
WEIGHT_VOLATILITY_SCORE = 0.5
BINARY_STATE_CHANGE_BONUS_SCORE = 2.0



def parse_sensor_code(sensor_code):
    sensor_type_abbr = ""
    unit_id_str = "UNKNOWN_UNIT"
    known_types = sorted(list(SENSOR_METADATA.keys()), key=len, reverse=True)
    for type_abbr_candidate in known_types:
        if sensor_code.startswith(type_abbr_candidate):
            sensor_type_abbr = type_abbr_candidate
            break
    if not sensor_type_abbr:
        sensor_type_abbr = "UNKNOWN_TYPE"
    if sensor_type_abbr != "UNKNOWN_TYPE" and len(sensor_code) > len(sensor_type_abbr):
        numeric_part_for_unit = sensor_code[len(sensor_type_abbr):]
        if numeric_part_for_unit and numeric_part_for_unit[0].isdigit():
            unit_id_str = numeric_part_for_unit[0]
    return sensor_type_abbr, unit_id_str, sensor_code


def get_sensor_full_info(sensor_code):
    type_abbr, unit_id_num, _ = parse_sensor_code(sensor_code)
    type_full_name = SENSOR_METADATA.get(type_abbr, "Unknown Sensor Type")
    unit_str = f"Unit {unit_id_num}" if unit_id_num != "UNKNOWN_UNIT" else "General Area"
    return type_full_name, unit_str, type_abbr


def get_simplified_trend(series_data):
    if len(series_data) < 2: return "stable", 0.0
    start_val, end_val = series_data[0], series_data[-1]
    diff_abs = end_val - start_val
    trend_score = diff_abs
    if abs(diff_abs) < TREND_DIFF_THRESHOLD_ABS: return "stable", trend_score
    return ("increasing", trend_score) if diff_abs > 0 else ("decreasing", trend_score)


def get_simplified_volatility(series_data):
    if len(series_data) < 2: return "low volatility", 0.0

    if np.isnan(series_data).any():

        pass

    data_range = np.nanmax(series_data) - np.nanmin(series_data)
    if np.isnan(data_range):
        return "unknown volatility", 0.0

    volatility_score = data_range
    if data_range < VOLATILITY_RANGE_THRESHOLDS['low']: return "low volatility", volatility_score
    if data_range < VOLATILITY_RANGE_THRESHOLDS['medium']: return "medium volatility", volatility_score
    return "high volatility", volatility_score


def get_binary_sensor_status(series_data, type_abbr):

    if type_abbr not in BINARY_TYPES:
        return "", False, 0.0

    if np.isnan(series_data).any():

        pass

    mean_val = np.nanmean(series_data)
    if np.isnan(mean_val):
        return "unknown state", False, 0.0

    state = "ON" if mean_val > 0.5 else "OFF"

    if type_abbr == "MV":
        state = "Open" if mean_val > 0.5 else "Closed"

    changes = 0
    if len(series_data) > 1:

        valid_series = series_data[~np.isnan(series_data)]
        if len(valid_series) > 1:
            changes = np.sum(np.abs(np.diff(valid_series)) > 0.5)

    changed_state_bool = changes > BINARY_CHANGE_THRESHOLD
    status_description = f"{state} and {'changed state' if changed_state_bool else 'remained stable'}"
    importance_bonus = BINARY_STATE_CHANGE_BONUS_SCORE if changed_state_bool else 0.0
    return status_description, changed_state_bool, importance_bonus



def generate_descriptive_text_for_window(window_data, sensor_headers):
    num_sensors = window_data.shape[1]
    all_sensor_details = []

    for i in range(num_sensors):
        header = sensor_headers[i]
        series = window_data[:, i]
        type_full, unit_str, type_abbr = get_sensor_full_info(header)

        is_binary_type = type_abbr in BINARY_TYPES
        binary_status_desc, changed_state_bool, binary_bonus_score = "", False, 0.0
        trend_desc, trend_score = "N/A", 0.0
        volatility_desc, volatility_score = "N/A", 0.0

        if is_binary_type:
            binary_status_desc, changed_state_bool, binary_bonus_score = get_binary_sensor_status(series, type_abbr)


            if not binary_status_desc.startswith("unknown"):
                trend_desc = "stable"
                volatility_desc = "low volatility"
            else:
                trend_desc = "unknown"
                volatility_desc = "unknown"
        else:
            trend_desc, trend_score = get_simplified_trend(series)
            volatility_desc, volatility_score = get_simplified_volatility(series)

        importance_score = (WEIGHT_TREND_SCORE * abs(trend_score) +
                            WEIGHT_VOLATILITY_SCORE * volatility_score +
                            binary_bonus_score)
        if "unknown" in trend_desc or "unknown" in volatility_desc or "unknown" in binary_status_desc:
            importance_score = -1

        all_sensor_details.append({
            "header": header, "type_full": type_full, "unit_str": unit_str, "type_abbr": type_abbr,
            "trend_desc": trend_desc, "volatility_desc": volatility_desc, "is_binary": is_binary_type,
            "binary_status_desc": binary_status_desc, "changed_state_binary": changed_state_bool,
            "importance_score": importance_score, "original_series_mean": np.nanmean(series)
        })

    valid_sensor_details = [s for s in all_sensor_details if s['importance_score'] != -1]
    if not valid_sensor_details:
        return "System status: Indeterminate due to data quality issues in this window."

    avg_importance = np.mean([s['importance_score'] for s in valid_sensor_details]) if valid_sensor_details else 0
    overall_status_desc = "System status: "
    if avg_importance < 1.0:
        overall_status_desc += "Appears mostly stable."
    elif avg_importance < 3.0:
        overall_status_desc += "Shows moderate activity."
    else:
        overall_status_desc += "Exhibits significant fluctuations."

    group_summary_parts = []

    unique_units = sorted(list(set(s['unit_str'] for s in valid_sensor_details if s['unit_str'] != "General Area")))
    groups_summarized_count = 0
    for unit in unique_units:
        if groups_summarized_count >= MAX_GROUP_SUMMARIES_DISPLAYED: break
        sensors_in_unit = [s for s in valid_sensor_details if s['unit_str'] == unit]
        if not sensors_in_unit: continue

        unit_max_importance = max(s['importance_score'] for s in sensors_in_unit) if sensors_in_unit else 0
        unit_desc_parts = []

        analog_sensors_in_unit = [s for s in sensors_in_unit if not s['is_binary']]
        if analog_sensors_in_unit:
            unit_trends = [s['trend_desc'] for s in analog_sensors_in_unit if "unknown" not in s['trend_desc']]
            if unit_trends:
                stable_ratio = unit_trends.count('stable') / len(unit_trends)
                increasing_ratio = unit_trends.count('increasing') / len(unit_trends)
                decreasing_ratio = unit_trends.count('decreasing') / len(unit_trends)
                if increasing_ratio >= GROUP_BEHAVIOR_THRESHOLD_RATIO:
                    unit_desc_parts.append("many sensors are increasing")
                elif decreasing_ratio >= GROUP_BEHAVIOR_THRESHOLD_RATIO:
                    unit_desc_parts.append("many sensors are decreasing")
                elif stable_ratio >= 0.8:
                    unit_desc_parts.append("most sensors are stable")

        binary_sensors_in_unit = [s for s in sensors_in_unit if s['is_binary']]
        if binary_sensors_in_unit:
            known_state_binary_sensors = [s for s in binary_sensors_in_unit if "unknown" not in s['binary_status_desc']]
            if known_state_binary_sensors:
                on_open_count = sum(1 for s in known_state_binary_sensors if
                                    "ON" in s['binary_status_desc'] or "Open" in s['binary_status_desc'])
                if on_open_count == len(known_state_binary_sensors):
                    unit_desc_parts.append("all relevant binary devices are ON/Open")
                elif on_open_count == 0:
                    unit_desc_parts.append("all relevant binary devices are OFF/Closed")
                else:
                    unit_desc_parts.append("binary devices in mixed states")
            if any(s['changed_state_binary'] for s in known_state_binary_sensors):
                unit_desc_parts.append("with some state changes noted")

        if unit_desc_parts and unit_max_importance > 1.5:
            group_summary_parts.append(f"In {unit}, {'; '.join(unit_desc_parts)}.")
            groups_summarized_count += 1
    group_summary_str = " ".join(group_summary_parts)

    sorted_sensors = sorted(valid_sensor_details, key=lambda x: x['importance_score'], reverse=True)
    top_n_detailed_descs = []
    for i in range(min(TOP_N_SENSORS, len(sorted_sensors))):
        sensor = sorted_sensors[i]
        if sensor['importance_score'] < 0.2 and i > 0: continue
        desc = f"{sensor['type_full']} {sensor['header']} from {sensor['unit_str']}"
        if sensor['is_binary']:
            desc += f" is {sensor['binary_status_desc']}."
        else:
            desc += f" is {sensor['trend_desc']} with {sensor['volatility_desc']}."
        top_n_detailed_descs.append(desc)

    top_n_summary_str = ""
    if top_n_detailed_descs:
        top_n_summary_str = "Notably: " + " ".join(top_n_detailed_descs)
    elif "stable" in overall_status_desc:
        top_n_summary_str = "All key sensors appear to be operating steadily."

    final_text_parts = [overall_status_desc]
    if group_summary_str: final_text_parts.append(group_summary_str)
    if top_n_summary_str: final_text_parts.append(top_n_summary_str)
    return " ".join(final_text_parts).strip().replace("  ", " ")



def process_csv_file_to_text_list(csv_file_path, window_size=50, verbose=True):
    all_generated_texts = []
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        if verbose: print(f"Error: CSV file not found at {csv_file_path}")
        return all_generated_texts
    except Exception as e:
        if verbose: print(f"Error reading CSV file: {e}")
        return all_generated_texts

    sensor_headers = df.columns.tolist()

    try:

        data_array = df.apply(pd.to_numeric, errors='coerce').to_numpy()
        if np.isnan(data_array).any() and verbose:
            print(
                f"Warning: CSV data contains non-numeric values which were converted to NaN. This will affect analysis.")
    except Exception as e:
        if verbose:
            print(f"Error: Could not convert CSV data to numeric array. Details: {e}")
        return all_generated_texts

    num_rows, num_cols = data_array.shape

    if num_rows < window_size:
        if verbose: print(
            f"Data has only {num_rows} rows, less than window size {window_size}. Cannot create any full windows.")
        return all_generated_texts

    if verbose:
        print(f"Loaded data: {num_rows} rows, {num_cols} sensors.")
        print(f"Sensor headers: {sensor_headers}")
        print(f"Window size: {window_size}\n")

    windows_processed = 0

    for i in range(0, num_rows, window_size):

        window_end = i + window_size


        if window_end > num_rows:
            if verbose:
                print(
                    f"--- Skipping last partial window (Rows {i + 1} to {num_rows}). Not enough data for a full window of size {window_size}. ---")
            break

        current_window_data = data_array[i: window_end, :]



        if verbose and (windows_processed < 3 or windows_processed % 100 == 0):
            print(f"--- Processing Window {windows_processed + 1} (Rows {i + 1} to {window_end}) ---")

        descriptive_text = generate_descriptive_text_for_window(current_window_data, sensor_headers)
        all_generated_texts.append(descriptive_text)

        if verbose and (windows_processed < 3 or windows_processed % 100 == 0):
            print("Generated Text (sample):")
            print(descriptive_text)
            print("-" * 30)
        windows_processed += 1

    if verbose:
        print(f"\nFinished processing. Generated text for {windows_processed} non-overlapping windows.")

    return all_generated_texts


if __name__ == "__main__":
    data_folder = r"../SWAT/train"
    csv_filename = "train_normal.csv"

    full_csv_path = os.path.join(data_folder, csv_filename)
    WINDOW_SIZE_FOR_PROCESSING = 50

    print(f"Attempting to process CSV: {full_csv_path}")

    generated_texts_list = process_csv_file_to_text_list(
        full_csv_path,
        window_size=WINDOW_SIZE_FOR_PROCESSING,
        verbose=True
    )

    if generated_texts_list:
        print(f"\nSuccessfully generated {len(generated_texts_list)} text descriptions.")

        print("\nFirst 5 generated texts:")
        for i, text in enumerate(generated_texts_list[:5]):
            print(f"Window {i + 1}: {text}")

        if len(generated_texts_list) > 5:
            print("\nLast 5 generated texts:")
            for i, text in enumerate(generated_texts_list[-5:],
                                     start=max(1, len(generated_texts_list) - 4)):
                print(f"Window {i}: {text}")

        output_text_file = os.path.join(data_folder, "generated_descriptions_non_overlapping.txt")
        try:
            with open(output_text_file, "w", encoding="utf-8") as f:
                for i, text in enumerate(generated_texts_list):
                    f.write(f"Window {i + 1}: {text}\n")
            print(f"\nAll generated texts saved to: {output_text_file}")
        except Exception as e:
            print(f"\nError saving texts to file: {e}")
    else:
        print("No text descriptions were generated. Please check for errors above.")

    print("\nReminder: The quality of the generated text heavily depends on the")
    print("threshold values and importance weights. These likely need to be tuned.")
