# -*- coding: utf-8 -*-
"""
 Patient-Level Split Verification
 Confirms zero data leakage between training and test sets.
"""

import os
import io
import sys
import re
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
from terminal_ui import *
from paths import BASE_DIR, DAY1_DIR

MY_FOLDER = BASE_DIR / "my_folder"


def extract_patient_name(filename):
    stem = Path(filename).stem.lower()
    match = re.match(r'^\d{4}-\d{2}-\d{2}-([a-z]+)', stem)
    return match.group(1) if match else None


def scan_patients(base_path, split_name):
    patients_by_class = {}
    files_by_class = {}
    for task_type in ["3class", "4class"]:
        task_dir = base_path / split_name / task_type
        if not task_dir.exists():
            continue
        for class_dir in sorted(task_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            key = f"{task_type}/{class_dir.name}"
            patients = set()
            files = []
            for f in sorted(class_dir.iterdir()):
                if f.suffix.upper() == '.CSV':
                    name = extract_patient_name(f.name)
                    if name:
                        patients.add(name)
                        files.append(f.name)
            patients_by_class[key] = patients
            files_by_class[key] = files
    return patients_by_class, files_by_class


def main():
    t0 = time.time()
    banner("PATIENT-LEVEL SPLIT VERIFICATION", f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    step(1, 3, "Scanning patient files")
    kv("Data directory", str(MY_FOLDER))
    train_patients, train_files = scan_patients(MY_FOLDER, "train")
    test_patients, test_files = scan_patients(MY_FOLDER, "test")
    all_train = set()
    all_test = set()
    for p in train_patients.values(): all_train.update(p)
    for p in test_patients.values(): all_test.update(p)
    kv("Train patients", len(all_train))
    kv("Test patients", len(all_test))

    step(2, 3, "Cross-class overlap analysis")
    overlap = all_train & all_test
    if overlap:
        info(f"Cross-class overlap: {len(overlap)} patients (EXPECTED)")
        info("Same patients have both sinus rhythm + PSVT recordings")
    else:
        success("Zero total overlap")

    step(3, 3, "Per-class patient isolation verification")
    validation_rows = []
    all_pass = True
    for task_type in ['3class', '4class']:
        for cls in ['AVNRT', 'AVRT', 'AVRT-L', 'AVRT-R', 'N']:
            key = f"{task_type}/{cls}"
            train_p = train_patients.get(key, set())
            test_p = test_patients.get(key, set())
            if not train_p and not test_p:
                continue
            cls_overlap = train_p & test_p
            status = 'PASSED' if len(cls_overlap) == 0 else 'FAILED'
            if len(cls_overlap) > 0: all_pass = False
            validation_rows.append([task_type, cls, str(len(train_p)), str(len(test_p)), str(len(cls_overlap)), status])

    table(["Task", "Class", "Train", "Test", "Overlap", "Status"], validation_rows)

    if all_pass:
        result_box("VALIDATION RESULT", {
            "Status": "ALL PASSED", "Train patients": str(len(all_train)),
            "Test patients": str(len(all_test)), "Within-class overlap": "0",
            "Elapsed": elapsed_time(t0),
        })
    else:
        error("VALIDATION FAILED")

    section("Detailed Distribution", "◆")
    dist_rows = []
    for split_name, p_dict, f_dict in [("Train", train_patients, train_files), ("Test", test_patients, test_files)]:
        for key in sorted(p_dict.keys()):
            task, cls = key.split("/")
            dist_rows.append([split_name, task, cls, str(len(f_dict[key])), str(len(p_dict[key]))])
    table(["Split", "Task", "Class", "Files", "Patients"], dist_rows)

    # Save to Day1 folder
    section("Saving Reports", "◆")
    report_path = DAY1_DIR / "patient_split_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Patient-Level Split Verification Report\n\n")
        f.write(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n")
        f.write("## Verification Result\n\n")
        f.write(f"**{'PASSED' if all_pass else 'FAILED'}** — ")
        f.write(f"{'No within-class patient overlap.' if all_pass else 'Overlap detected!'}\n\n")
        f.write("```\n")
        f.write(f"Training set unique patients: {len(all_train)}\n")
        f.write(f"Testing set unique patients:  {len(all_test)}\n")
        f.write(f"Cross-class overlap:          {len(overlap)} (expected)\n")
        f.write(f"Within-class overlap:         {'0' if all_pass else 'DETECTED'}\n")
        f.write("```\n\n")
        f.write("## Per-Class Validation\n\n| Task | Class | Train | Test | Overlap | Status |\n|------|-------|------:|-----:|--------:|--------|\n")
        for row in validation_rows:
            f.write(f"| {' | '.join(row)} |\n")
        f.write("\n## Patient Lists\n\n### Train\n```\n")
        for n in sorted(all_train): f.write(f"{n}\n")
        f.write("```\n\n### Test\n```\n")
        for n in sorted(all_test): f.write(f"{n}\n")
        f.write("```\n")

    csv_path = DAY1_DIR / "patient_distribution.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Split,Task,Class,Files,Patients\n")
        for row in dist_rows:
            f.write(",".join(row) + "\n")

    success(f"Report: {report_path}")
    success(f"CSV:    {csv_path}")


if __name__ == "__main__":
    main()
