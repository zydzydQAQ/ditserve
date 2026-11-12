# SLO Configuration Files

This directory contains SLO (Service Level Objective) configuration files for the TetriServe serving benchmark.

## How It Works

Each JSON file defines SLO values for different batch sizes and image resolutions.

**Two formats supported:**

### **1. Multiplier Format** (relative to baseline):
```json
{
  "(1, 256, 256)": "2.0x",   // 2.0 × baseline latency
  "(2, 256, 256)": "1.8x",   // 1.8 × baseline latency
  "(4, 256, 256)": "1.5x"    // 1.5 × baseline latency
}
```

### **2. Fixed Value Format** (absolute deadlines):
```json
{
  "(1, 512, 512)": 15.0,     // Exactly 15 seconds
  "(2, 512, 512)": 25.0,     // Exactly 25 seconds
  "(4, 512, 512)": 35.0      // Exactly 35 seconds
}
```

### **3. Mixed Format** (both in same file):
```json
{
  "(1, 256, 256)": "2.0x",   // Multiplier
  "(2, 256, 256)": 15.0,     // Fixed value
  "(4, 256, 256)": "1.5x",   // Multiplier
  "(8, 256, 256)": 25.0      // Fixed value
}
```

**How it works:**
- **Multipliers ("2.0x")**: Final SLO = Baseline SLO × Multiplier
- **Fixed values (15.0)**: Final SLO = Fixed value (ignores baseline)
- Baseline SLO loaded from scaling efficiency data

## Usage

In your benchmark configuration file (YAML), specify which SLO configs to use:

```yaml
# Use a single SLO configuration
slo_configs: [default]

# Test multiple SLO configurations in the same benchmark
slo_configs: [tight, default, relaxed]
```

## Available Configurations

- **`default.json`**: Reasonable SLO multipliers (1.3x-2.0x baseline)

## Creating Custom SLO Configs

1. Create a new JSON file in this directory
2. Define SLO multipliers for each (batch_size, height, width) combination
3. Reference it in your benchmark config:

```yaml
slo_configs: [my_custom_slo]
```

Example custom config:
```json
{
  "(1, 512, 512)": 1.5,
  "(2, 512, 512)": 1.3,
  "(4, 512, 512)": 1.1,
  "(1, 1024, 1024)": 2.5,
  "(2, 1024, 1024)": 2.2,
  "(4, 1024, 1024)": 1.8
}
```

## Key Format

- Use `"(batch_size, height, width)"` format with parentheses and quotes
- Include all batch sizes your benchmark will use (1, 2, 4, 8)
- Multiplier values: <1.0 for tighter SLOs, >1.0 for looser SLOs
- System falls back to base SLO if no matching key found
