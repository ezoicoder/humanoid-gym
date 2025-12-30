#!/usr/bin/env python3
"""
Simple test for Stage 1 virtual terrain generation function.
"""

# Import isaacgym before torch
from isaacgym import gymapi, terrain_utils
import numpy as np
import sys
sys.path.insert(0, '/workspace_robot/humanoid-gym')

# Import terrain generation function directly
from humanoid.utils.terrain import stones_everywhere_stage1_terrain

def test_stage1_terrain_function():
    """Test the stones_everywhere_stage1_terrain function directly."""
    print("\n" + "="*70)
    print("TEST: stones_everywhere_stage1_terrain Function")
    print("="*70)
    
    # Create a SubTerrain object
    width_pixels = 425  # 8.5m / 0.02m
    length_pixels = 425
    horizontal_scale = 0.02
    vertical_scale = 0.005
    
    terrain = terrain_utils.SubTerrain(
        "test_terrain",
        width=width_pixels,
        length=length_pixels,
        vertical_scale=vertical_scale,
        horizontal_scale=horizontal_scale
    )
    
    print(f"Created SubTerrain: {width_pixels}x{length_pixels} pixels")
    print(f"  Horizontal scale: {horizontal_scale}m")
    print(f"  Vertical scale: {vertical_scale}m")
    
    # Test different difficulty levels
    for difficulty in [0.0, 0.5, 1.0]:
        print(f"\n--- Testing difficulty = {difficulty} ---")
        
        # Generate Stage 1 terrain
        stones_everywhere_stage1_terrain(terrain, difficulty)
        
        # Check that both height fields exist
        assert hasattr(terrain, 'height_field_raw'), "height_field_raw not found!"
        assert hasattr(terrain, 'height_field_virtual'), "height_field_virtual not found!"
        print("✓ Both height_field_raw and height_field_virtual exist")
        
        # Analyze physical terrain (should be flat)
        physical_unique = np.unique(terrain.height_field_raw)
        physical_mean = np.mean(terrain.height_field_raw)
        physical_std = np.std(terrain.height_field_raw)
        print(f"  Physical terrain:")
        print(f"    - Unique values: {len(physical_unique)}")
        print(f"    - Mean: {physical_mean:.2f}, Std: {physical_std:.2f}")
        print(f"    - Is flat: {physical_std < 1.0}")
        
        # Analyze virtual terrain (should have stones)
        virtual_unique = np.unique(terrain.height_field_virtual)
        virtual_mean = np.mean(terrain.height_field_virtual)
        virtual_std = np.std(terrain.height_field_virtual)
        virtual_min = np.min(terrain.height_field_virtual)
        virtual_max = np.max(terrain.height_field_virtual)
        
        print(f"  Virtual terrain:")
        print(f"    - Unique values: {len(virtual_unique)}")
        print(f"    - Mean: {virtual_mean:.2f}, Std: {virtual_std:.2f}")
        print(f"    - Range: [{virtual_min}, {virtual_max}]")
        print(f"    - Has variation: {virtual_std > 10.0}")
        
        # Count stones vs pits
        stones = np.sum(terrain.height_field_virtual > -10)
        pits = np.sum(terrain.height_field_virtual < -10)
        total = terrain.height_field_virtual.size
        print(f"    - Stones: {stones}/{total} ({stones/total*100:.1f}%)")
        print(f"    - Pits: {pits}/{total} ({pits/total*100:.1f}%)")
        
        # Verify expectations
        assert physical_std < 1.0, f"Physical terrain should be flat! std={physical_std}"
        assert virtual_std > 10.0, f"Virtual terrain should have variation! std={virtual_std}"
        print("✓ Terrain properties are correct")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("""
Stage 1 Virtual Terrain Implementation Verified!

The stones_everywhere_stage1_terrain function correctly generates:
  ✓ Flat physical terrain (height_field_raw = 0)
  ✓ Stones virtual terrain (height_field_virtual with stones pattern)

This enables safe Stage 1 training where:
  - Robot walks on flat ground (no falls)
  - Robot perceives stones terrain (elevation map)
  - Foothold reward based on virtual stones
    """)

if __name__ == "__main__":
    try:
        test_stage1_terrain_function()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

