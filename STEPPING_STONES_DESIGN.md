# Stepping Stones Terrain - Technical Design Document

## Overview
Stepping Stones is a new challenging terrain type that requires precise foot placement across alternating stepping stones, similar to Balancing Beams but with discrete stone platforms instead of continuous beams.

## Terrain Specifications

### Dimensions
- **Total terrain size**: 8.5m × 8.5m
- **Effective area**: 2m (width) × 8m (length) in the center
- **Border**: 0.25m on all sides (height = 0)
- **Outside effective area**: Deep pit (1.0m depth)

### Platform Layout
- **Start platform**: 1.5m × 1m at the beginning
- **End platform**: 1.5m × 1m at the end
- **Stone arrangement**: Two alternating rows (left-right zigzag pattern)

### Difficulty Parameters (l ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8})

Mapped from continuous difficulty ∈ [0, 1] to discrete level l:
- difficulty 0.0 → l=0 (easiest)
- difficulty 1.0 → l=8 (hardest)

#### Stone Size Sequence
```
l:          0     1     2     3     4     5     6     7     8
stone_size: 0.8   0.65  0.5   0.4   0.35  0.3   0.25  0.2   0.2  (meters)
```

#### Stone Distance (Forward Spacing)
Formula: `stone_distance = 0.1 + 0.05 * l` meters

```
l:              0     1     2     3     4     5     6      7      8
stone_distance: 0.10  0.15  0.20  0.25  0.30  0.35  0.40   0.45   0.50  (meters)
```

#### Lateral Offset Pattern
- Stones alternate between left and right sides
- Left stone: x_center = mid_x - 0.4m
- Right stone: x_center = mid_x + 0.4m
- Fixed lateral spacing for all difficulty levels

### Height Variation
Each stone has random height variation: ±0.05m (as per paper specification)

## Robot Spawn Position
**Type 9 - Stepping Stones**:
- Spawn on start platform (y = -3.5m relative to terrain center)
- Lateral offset: x ∈ [-0.3m, +0.3m] for variation
- Ensures robot starts on solid ground

## Success Criteria
**Success condition**: Robot reaches end platform
- Success detection: `rel_pos[:, 0] > 1.0` (similar to Balancing Beams)
- Success only sets flag, does not terminate episode
- Used for terrain curriculum progression

## Coordinate System
- **x-direction**: Lateral (left-right, where stones alternate)
- **y-direction**: Forward (from start platform to end platform)
- **z-direction**: Height (with ±0.05m variation)

## Visual Comparison with Other Terrains

### Balancing Beams (Type 7)
- Continuous beams with periodic lateral offset
- Stone width varies with difficulty
- Zigzag pattern following sine wave

### Stepping Stones (Type 9) - NEW
- Discrete square stones
- Two rows alternating left-right
- Stone size decreases, spacing increases with difficulty
- More challenging foot placement required

### Stones Everywhere (Type 8)
- 2D grid of stones across entire 8m × 8m area
- No specific path, exploration required
- Central 4m × 4m platform as safe zone

## Implementation Files

### 1. terrain.py
- Add `stepping_stones_terrain(terrain, difficulty)` function
- Register terrain type 9 in `HumanoidTerrain.make_terrain()`

### 2. humanoid_env.py
- Add Type 9 spawn logic in `_reset_root_states()`
- Add Type 9 success detection in `_check_success_criteria()`

### 3. play_terrain_curriculum.py
- Update `terrain_proportions` to include new terrain type
- Adjust `num_cols` to accommodate 3 terrain types

## Terrain Type Mapping
```
Type 0: flat
Type 1: discrete obstacles
Type 2: random uniform
Type 3: pyramid slope (positive)
Type 4: pyramid slope (negative)
Type 5: pyramid stairs (up)
Type 6: pyramid stairs (down)
Type 7: balancing beams
Type 8: stones everywhere
Type 9: stepping stones (NEW)
```

## Expected Behavior
1. **Easy difficulty (l=0,1,2)**: Large stones (0.5-0.8m), close spacing (0.1-0.2m) → walkable
2. **Medium difficulty (l=3,4,5)**: Medium stones (0.3-0.4m), moderate spacing (0.25-0.35m) → requires balance
3. **Hard difficulty (l=6,7,8)**: Small stones (0.2-0.25m), wide gaps (0.4-0.5m) → requires jumping/precise control

## Testing Recommendations
1. Verify terrain generation produces correct stone sizes and spacing
2. Test robot spawning on start platform
3. Verify success detection at end platform
4. Test curriculum progression (upgrade on success, downgrade on failure)
5. Compare performance across all three terrain types (beams, stones_everywhere, stepping_stones)

---

## Implementation Status: ✅ COMPLETED

### Files Modified

#### 1. `humanoid/utils/terrain.py`
**Changes:**
- Added `stepping_stones_terrain(terrain, difficulty)` function (lines 482-599)
  - Implements stone generation with alternating left-right pattern
  - Stone sizes follow specification: [0.8, 0.65, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2]
  - Stone distance formula: 0.1 + 0.05*l meters
  - Creates 8.5m × 8.5m terrain with 2m effective width
  - Generates start and end platforms (1.5m × 1m each)
  - Surrounding area is deep pit (-1.0m)
  
- Updated `HumanoidTerrain.make_terrain()` to handle Type 9
  - Added terrain_type = 9 for stepping stones
  - Updated proportions logic to support 3 terrain types

#### 2. `humanoid/envs/custom/humanoid_env.py`
**Changes:**
- Updated `_init_terrain_types()` terrain_names dictionary to include Type 9
  - Added `9: 'stepping_stones'` mapping
  
- Updated `_reset_root_states()` method
  - Type 9 now shares spawn logic with Type 7 (Balancing Beams)
  - Robots spawn on start platform at y = -3.5m relative to terrain center
  - Lateral randomization: x ∈ [-0.3m, +0.3m]
  
- Updated `_check_success_criteria()` method
  - Type 9 shares success criteria with Type 7
  - Success condition: `rel_pos[:, 0] > 1.0` (reached end platform)
  - Added Type 9 to success tracking statistics
  - Logs `stepping_stones_success_count` to extras

#### 3. `humanoid/scripts/play_terrain_curriculum.py`
**Changes:**
- Updated `num_cols = 3` (previously 2) to accommodate 3 terrain types
- Updated `terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0.33, 0.67, 1.0]`
  - Column 0: Balancing Beams (0.0 - 0.33]
  - Column 1: Stones Everywhere (0.33 - 0.67]
  - Column 2: Stepping Stones (0.67 - 1.0]
- Updated robot distribution print statements
  - Now displays all 3 terrain types
  - Shows correct terrain name for each robot

### How to Use

#### Running the Simulation
```bash
cd /workspace_robot/humanoid-gym
python humanoid/scripts/play_terrain_curriculum.py --task=<your_task_name>
```

#### Configuration Options
In `play_terrain_curriculum.py`, you can adjust:
- `num_rows`: Number of difficulty levels (default: 5)
- `num_cols`: Number of terrain types (default: 3)
- `terrain_proportions`: Distribution of terrain types

#### Expected Output
When running, you should see:
```
Distributing robots across terrain curriculum (5 difficulty × 3 terrain types)...
Terrain layout (uniform 8.5m × 8.5m per cell):
  Column 0: Balancing Beams (2m width × 8m length effective, 1.5m×1m platforms)
  Column 1: Stones Everywhere (8m × 8m effective, 4m×4m central platform)
  Column 2: Stepping Stones (2m width × 8m length effective, 1.5m×1m platforms, alternating stones)
  Rows: 5 difficulty levels from 0.0 to 1.0
```

And for each terrain cell:
```
[Stepping Stones] difficulty=0.00, l=0, stone_size=0.800m, stone_distance=0.100m, lateral_offset=0.400m, effective_width=2.0m
[Stepping Stones] difficulty=0.25, l=2, stone_size=0.500m, stone_distance=0.200m, lateral_offset=0.400m, effective_width=2.0m
...
```

### Verification Checklist
- [x] Terrain generation function implemented
- [x] Terrain type registered in HumanoidTerrain
- [x] Robot spawn position configured
- [x] Success criteria implemented
- [x] Play script updated for 3 terrain types
- [x] No linting errors
- [x] Documentation complete

### Notes
1. **Stone Pattern**: Stones alternate left-right with fixed lateral offset of 0.4m from center
2. **Difficulty Progression**: As difficulty increases (l: 0→8):
   - Stones get smaller: 0.8m → 0.2m
   - Gaps get larger: 0.1m → 0.5m
3. **Comparison with Balancing Beams**:
   - Beams: Continuous platforms with variable width
   - Stepping: Discrete stones with fixed size per difficulty level
   - Stepping requires more precise foot placement

### Future Enhancements
1. Add visualization of success rates per terrain type
2. Implement dynamic difficulty adjustment based on robot performance
3. Add more terrain variants (e.g., random stone heights, irregular patterns)
4. Create mixed terrain curriculum combining multiple types in sequence

