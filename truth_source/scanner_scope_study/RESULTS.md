# Scanner scope study

modelscan 0.8.6, picklescan 1.0.4. Panel A is the positive control: if those rows are not FLAGGED, the harness is broken and the rest is void.

| panel | artifact                            | file                        | bytes     | modelscan     | scanner used       | streams | severity                 | picklescan    |
| ----- | ----------------------------------- | --------------------------- | --------- | ------------- | ------------------ | ------- | ------------------------ | ------------- |
| A     | `control/unsafe pickle (os.system)` | `control_unsafe.pkl`        | 60        | **FLAGGED**   | PickleUnsafeOpScan | 1       | CRITICAL,HIGH,MEDIUM,LOW | **FLAGGED**   |
| A     | `control/unsafe payload in .pt`     | `control_unsafe.pt`         | 892       | **FLAGGED**   | PickleUnsafeOpScan | 1       | CRITICAL,HIGH,MEDIUM,LOW | **FLAGGED**   |
| A     | `control/benign pickle`             | `control_benign.pkl`        | 70        | **no issues** | PickleUnsafeOpScan | 1       | -                        | **no issues** |
| B     | `clean/.pth state_dict`             | `clean_state_dict.pth`      | 19800     | **no issues** | PickleUnsafeOpScan | 1       | -                        | **no issues** |
| B     | `clean/.pt full pickle`             | `clean_full.pt`             | 20712     | **no issues** | PickleUnsafeOpScan | 1       | -                        | **no issues** |
| B     | `clean/.ckpt`                       | `clean_ckpt.ckpt`           | 19752     | **no issues** | PickleUnsafeOpScan | 1       | -                        | **no issues** |
| B     | `clean/.bin`                        | `clean_weights.bin`         | 19776     | **no issues** | PickleUnsafeOpScan | 1       | -                        | **no issues** |
| B     | `clean/.pt TorchScript`             | `clean_script.pt`           | 24128     | **no issues** | PickleUnsafeOpScan | 3       | -                        | **no issues** |
| B     | `backdoored/.pth state_dict`        | `backdoored_state_dict.pth` | 37864     | **no issues** | PickleUnsafeOpScan | 1       | -                        | **no issues** |
| B     | `backdoored/.pt full pickle`        | `backdoored_full.pt`        | 39008     | **no issues** | PickleUnsafeOpScan | 1       | -                        | **no issues** |
| B     | `backdoored/.ckpt`                  | `backdoored_ckpt.ckpt`      | 37792     | **no issues** | PickleUnsafeOpScan | 1       | -                        | **no issues** |
| B     | `backdoored/.bin`                   | `backdoored_weights.bin`    | 37828     | **no issues** | PickleUnsafeOpScan | 1       | -                        | **no issues** |
| B     | `backdoored/.pt TorchScript`        | `backdoored_script.pt`      | 44945     | **no issues** | PickleUnsafeOpScan | 3       | -                        | **no issues** |
| C     | `clean/toy add-DGP`                 | `clean_toy.onnx`            | 18124     | **no issues** | none               | 0       | -                        | **no issues** |
| C     | `backdoored/toy add-DGP`            | `backdoored_toy.onnx`       | 35577     | **no issues** | none               | 0       | -                        | **no issues** |
| C     | `clean/TensorFlow-authored`         | `clean_tf.onnx`             | 14553     | **no issues** | none               | 0       | -                        | **no issues** |
| C     | `backdoored/TensorFlow-authored`    | `backdoored_tf.onnx`        | 15880     | **no issues** | none               | 0       | -                        | **no issues** |
| C     | `backdoored/benchmark op_sep_tar`   | `op_sep_tar_default.onnx`   | 44675712  | **no issues** | none               | 0       | -                        | **no issues** |
| C     | `clean/benchmark resnet50`          | `resnet50.onnx`             | 102057645 | **no issues** | none               | 0       | -                        | **no issues** |
| C     | `backdoored/production GPT-J-6B`    | `gpt-j-6b-backdoored.onnx`  | 1186776   | **no issues** | none               | 0       | -                        | **no issues** |

TorchScript artifact contains the gate operators (relu, mul, add): **True**

## Panel D: ArchProof on the same ONNX files

| artifact                          | verdict                    | epsilon |
| --------------------------------- | -------------------------- | ------- |
| `clean/toy add-DGP`               | add-DGP-CLASS-NEGATIVE     | 0       |
| `backdoored/toy add-DGP`          | add-DGP-CERTIFIED-POSITIVE | 13740.4 |
| `clean/TensorFlow-authored`       | add-DGP-CLASS-NEGATIVE     | 0       |
| `backdoored/TensorFlow-authored`  | add-DGP-CERTIFIED-POSITIVE | 36      |
| `backdoored/benchmark op_sep_tar` | add-DGP-CERTIFIED-POSITIVE | 1       |
| `clean/benchmark resnet50`        | add-DGP-CLASS-NEGATIVE     | 0       |

## Raw scanner output

### [A] control/unsafe pickle (os.system)

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 

Scanning truth_source/scanner_scope_study/control_unsafe.pkl using modelscan.scanners.PickleUnsafeOpScan model scan

--- Summary ---

Total Issues: 1

Total Issues By Severity:

    - LOW: 0
    - MEDIUM: 0
    - HIGH: 0
    - CRITICAL: 1

--- Issues by Severity ---

--- CRITICAL ---

Unsafe operator found:
  - Severity: CRITICAL
  - Description: Use of unsafe operator 'system' from module 'posix'
  - Source: truth_source/scanner_scope_study/control_unsafe.pkl
```

**picklescan**

```
truth_source/scanner_scope_study/control_unsafe.pkl: dangerous import 'posix system' FOUND
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 1
Dangerous globals: 1
```

### [A] control/unsafe payload in .pt

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 

Scanning truth_source/scanner_scope_study/control_unsafe.pt:control_unsafe/data.pkl using modelscan.scanners.PickleUnsafeOpScan model scan

--- Summary ---

Total Issues: 1

Total Issues By Severity:

    - LOW: 0
    - MEDIUM: 0
    - HIGH: 0
    - CRITICAL: 1

--- Issues by Severity ---

--- CRITICAL ---

Unsafe operator found:
  - Severity: CRITICAL
  - Description: Use of unsafe operator 'system' from module 'posix'
  - Source: truth_source/scanner_scope_study/control_unsafe.pt:control_unsafe/data.pkl

--- Skipped --- 

Total skipped: 3 - run with --show-skipped to see the full list.
```

**picklescan**

```
truth_source/scanner_scope_study/control_unsafe.pt:control_unsafe/data.pkl: dangerous import 'posix system' FOUND
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 1
Dangerous globals: 1
```

### [A] control/benign pickle

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 

Scanning truth_source/scanner_scope_study/control_benign.pkl using modelscan.scanners.PickleUnsafeOpScan model scan

--- Summary ---

 No issues found! 🎉
```

**picklescan**

```
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 0
Dangerous globals: 0
```

### [B] clean/.pth state_dict

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 

Scanning truth_source/scanner_scope_study/clean_state_dict.pth:clean_state_dict/data.pkl using modelscan.scanners.PickleUnsafeOpScan model scan

--- Summary ---

 No issues found! 🎉

--- Skipped --- 

Total skipped: 7 - run with --show-skipped to see the full list.
```

**picklescan**

```
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 0
Dangerous globals: 0
```

### [B] clean/.pt full pickle

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 

Scanning truth_source/scanner_scope_study/clean_full.pt:clean_full/data.pkl using modelscan.scanners.PickleUnsafeOpScan model scan

--- Summary ---

 No issues found! 🎉

--- Skipped --- 

Total skipped: 7 - run with --show-skipped to see the full list.
```

**picklescan**

```
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 0
Dangerous globals: 0
```

### [B] clean/.ckpt

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 

Scanning truth_source/scanner_scope_study/clean_ckpt.ckpt:clean_ckpt/data.pkl using modelscan.scanners.PickleUnsafeOpScan model scan

--- Summary ---

 No issues found! 🎉

--- Skipped --- 

Total skipped: 7 - run with --show-skipped to see the full list.
```

**picklescan**

```
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 0
Dangerous globals: 0
```

### [B] clean/.bin

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 

Scanning truth_source/scanner_scope_study/clean_weights.bin:clean_weights/data.pkl using modelscan.scanners.PickleUnsafeOpScan model scan

--- Summary ---

 No issues found! 🎉

--- Skipped --- 

Total skipped: 7 - run with --show-skipped to see the full list.
```

**picklescan**

```
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 0
Dangerous globals: 0
```

### [B] clean/.pt TorchScript

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 

Scanning truth_source/scanner_scope_study/clean_script.pt:clean_script/data.pkl using modelscan.scanners.PickleUnsafeOpScan model scan
Scanning truth_source/scanner_scope_study/clean_script.pt:clean_script/constants.pkl using modelscan.scanners.PickleUnsafeOpScan model scan
Scanning truth_source/scanner_scope_study/clean_script.pt:clean_script/traced_inputs.pkl using modelscan.scanners.PickleUnsafeOpScan model scan

--- Summary ---

 No issues found! 🎉

--- Skipped --- 

Total skipped: 14 - run with --show-skipped to see the full list.
```

**picklescan**

```
----------- SCAN SUMMARY -----------
Scanned files: 6
Infected files: 0
Dangerous globals: 0
```

### [B] backdoored/.pth state_dict

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 

Scanning truth_source/scanner_scope_study/backdoored_state_dict.pth:backdoored_state_dict/data.pkl using modelscan.scanners.PickleUnsafeOpScan model scan

--- Summary ---

 No issues found! 🎉

--- Skipped --- 

Total skipped: 11 - run with --show-skipped to see the full list.
```

**picklescan**

```
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 0
Dangerous globals: 0
```

### [B] backdoored/.pt full pickle

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 

Scanning truth_source/scanner_scope_study/backdoored_full.pt:backdoored_full/data.pkl using modelscan.scanners.PickleUnsafeOpScan model scan

--- Summary ---

 No issues found! 🎉

--- Skipped --- 

Total skipped: 11 - run with --show-skipped to see the full list.
```

**picklescan**

```
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 0
Dangerous globals: 0
```

### [B] backdoored/.ckpt

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 

Scanning truth_source/scanner_scope_study/backdoored_ckpt.ckpt:backdoored_ckpt/data.pkl using modelscan.scanners.PickleUnsafeOpScan model scan

--- Summary ---

 No issues found! 🎉

--- Skipped --- 

Total skipped: 11 - run with --show-skipped to see the full list.
```

**picklescan**

```
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 0
Dangerous globals: 0
```

### [B] backdoored/.bin

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 

Scanning truth_source/scanner_scope_study/backdoored_weights.bin:backdoored_weights/data.pkl using modelscan.scanners.PickleUnsafeOpScan model scan

--- Summary ---

 No issues found! 🎉

--- Skipped --- 

Total skipped: 11 - run with --show-skipped to see the full list.
```

**picklescan**

```
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 0
Dangerous globals: 0
```

### [B] backdoored/.pt TorchScript

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 

Scanning truth_source/scanner_scope_study/backdoored_script.pt:backdoored_script/data.pkl using modelscan.scanners.PickleUnsafeOpScan model scan
Scanning truth_source/scanner_scope_study/backdoored_script.pt:backdoored_script/constants.pkl using modelscan.scanners.PickleUnsafeOpScan model scan
Scanning truth_source/scanner_scope_study/backdoored_script.pt:backdoored_script/traced_inputs.pkl using modelscan.scanners.PickleUnsafeOpScan model scan

--- Summary ---

 No issues found! 🎉

--- Skipped --- 

Total skipped: 22 - run with --show-skipped to see the full list.
```

**picklescan**

```
----------- SCAN SUMMARY -----------
Scanned files: 8
Infected files: 0
Dangerous globals: 0
```

### [C] clean/toy add-DGP

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 


--- Summary ---

 No issues found! 🎉

--- Skipped --- 

Total skipped: 1 - run with --show-skipped to see the full list.
```

**picklescan**

```
WARNING: could not parse truth_source/scanner_scope_study/clean_toy.onnx as pickle: at position 0, opcode b'\x08' unknown
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 0
Dangerous globals: 0
```

### [C] backdoored/toy add-DGP

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 


--- Summary ---

 No issues found! 🎉

--- Skipped --- 

Total skipped: 1 - run with --show-skipped to see the full list.
```

**picklescan**

```
WARNING: could not parse truth_source/scanner_scope_study/backdoored_toy.onnx as pickle: at position 0, opcode b'\x08' unknown
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 0
Dangerous globals: 0
```

### [C] clean/TensorFlow-authored

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 


--- Summary ---

 No issues found! 🎉

--- Skipped --- 

Total skipped: 1 - run with --show-skipped to see the full list.
```

**picklescan**

```
WARNING: could not parse truth_source/crossformat_tf_onnx_demo/clean_tf.onnx as pickle: at position 0, opcode b'\x08' unknown
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 0
Dangerous globals: 0
```

### [C] backdoored/TensorFlow-authored

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 


--- Summary ---

 No issues found! 🎉

--- Skipped --- 

Total skipped: 1 - run with --show-skipped to see the full list.
```

**picklescan**

```
WARNING: could not parse truth_source/crossformat_tf_onnx_demo/backdoored_tf.onnx as pickle: at position 0, opcode b'\x08' unknown
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 0
Dangerous globals: 0
```

### [C] backdoored/benchmark op_sep_tar

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 


--- Summary ---

 No issues found! 🎉

--- Skipped --- 

Total skipped: 1 - run with --show-skipped to see the full list.
```

**picklescan**

```
WARNING: could not parse benchmark/exporter_test/op_sep_tar_default.onnx as pickle: at position 0, opcode b'\x08' unknown
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 0
Dangerous globals: 0
```

### [C] clean/benchmark resnet50

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 


--- Summary ---

 No issues found! 🎉

--- Skipped --- 

Total skipped: 1 - run with --show-skipped to see the full list.
```

**picklescan**

```
WARNING: could not parse benchmark/clean/resnet50.onnx as pickle: at position 0, opcode b'\x08' unknown
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 0
Dangerous globals: 0
```

### [C] backdoored/production GPT-J-6B

**modelscan**

```
No settings file detected at modelscan-settings.toml. Using defaults. 


--- Summary ---

 No issues found! 🎉

--- Skipped --- 

Total skipped: 1 - run with --show-skipped to see the full list.
```

**picklescan**

```
WARNING: could not parse benchmark/7b_onnx/backdoored_onnx/gpt-j-6b-backdoored/gpt-j-6b-backdoored.onnx as pickle: at position 0, opcode b'\x08' unknown
----------- SCAN SUMMARY -----------
Scanned files: 1
Infected files: 0
Dangerous globals: 0
```
