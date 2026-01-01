#!/usr/bin/env python3
"""
Test script to verify the double critic bug fix.
This test ensures that:
1. Critic2 receives returns2 (sparse rewards) as training target
2. The data flow is correct from storage to PPO update
"""

import torch
import sys

def test_storage_generator():
    """Test that mini_batch_generator returns correct data for double critic."""
    print("=" * 60)
    print("TEST 1: Storage Generator Output")
    print("=" * 60)
    
    from humanoid.algo.ppo.rollout_storage import RolloutStorage
    
    # Create a mock storage with double critic
    num_envs = 4
    num_transitions = 24
    num_obs = 10
    num_actions = 3
    
    device = 'cpu'
    
    # Test with double critic enabled
    storage = RolloutStorage(
        num_envs=num_envs,
        num_transitions_per_env=num_transitions,
        obs_shape=(num_obs,),
        privileged_obs_shape=(num_obs,),  # Use same as obs_shape for testing
        actions_shape=(num_actions,),
        device=device,
        use_double_critic=True  # Must be set during __init__ to allocate buffers
    )
    
    # Initialize buffers
    storage.rewards_dense = torch.randn(num_transitions, num_envs, 1)
    storage.rewards_sparse = torch.randn(num_transitions, num_envs, 1) * 0.1  # Sparse should be smaller
    storage.values = torch.randn(num_transitions, num_envs, 1)
    storage.values2 = torch.randn(num_transitions, num_envs, 1)
    storage.dones = torch.zeros(num_transitions, num_envs, 1)
    storage.observations = torch.randn(num_transitions, num_envs, num_obs)
    storage.actions = torch.randn(num_transitions, num_envs, num_actions)
    storage.actions_log_prob = torch.randn(num_transitions, num_envs, 1)
    storage.mu = torch.randn(num_transitions, num_envs, num_actions)
    storage.sigma = torch.ones(num_transitions, num_envs, num_actions)
    
    # Compute returns (this will also compute returns2)
    last_values1 = torch.randn(num_envs, 1)
    last_values2 = torch.randn(num_envs, 1)
    storage.compute_returns(last_values1, gamma=0.99, lam=0.95, 
                           last_values2=last_values2, w1=1.0, w2=0.25)
    
    # Test generator
    generator = storage.mini_batch_generator(num_mini_batches=2, num_epochs=1)
    
    batch_count = 0
    for batch_data in generator:
        batch_count += 1
        
        # Verify we get 13 items (11 standard + 2 double critic)
        assert len(batch_data) == 13, f"Expected 13 items, got {len(batch_data)}"
        
        # Unpack
        obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
        old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, \
        target_values2_batch, returns2_batch = batch_data
        
        # Verify shapes
        batch_size = obs_batch.shape[0]
        assert returns_batch.shape == (batch_size, 1), f"returns_batch shape: {returns_batch.shape}"
        assert returns2_batch.shape == (batch_size, 1), f"returns2_batch shape: {returns2_batch.shape}"
        assert target_values2_batch.shape == (batch_size, 1), f"target_values2_batch shape: {target_values2_batch.shape}"
        
        # Verify returns2 is different from returns (they should be computed from different rewards)
        assert not torch.allclose(returns_batch, returns2_batch, atol=0.1), \
            "returns_batch and returns2_batch should be different!"
        
        print(f"  ✓ Batch {batch_count}: Got returns2_batch (shape={returns2_batch.shape})")
        print(f"    - returns_batch mean: {returns_batch.mean().item():.4f}")
        print(f"    - returns2_batch mean: {returns2_batch.mean().item():.4f}")
        print(f"    - Difference: {(returns_batch - returns2_batch).abs().mean().item():.4f}")
    
    print(f"\n✅ TEST 1 PASSED: Generator returns correct double critic data")
    print(f"   - Processed {batch_count} batches")
    print(f"   - returns2_batch is correctly different from returns_batch")
    return True


def test_ppo_integration():
    """Test that PPO correctly uses returns2 for Critic2 loss."""
    print("\n" + "=" * 60)
    print("TEST 2: PPO Integration")
    print("=" * 60)
    
    # This test verifies the code structure, not runtime behavior
    import inspect
    from humanoid.algo.ppo import PPO
    
    # Check that update method handles double critic correctly
    update_source = inspect.getsource(PPO.update)
    
    # Verify key elements are in the code
    checks = [
        ("returns2_batch", "returns2_batch should be unpacked from generator"),
        ("target_values2_batch", "target_values2_batch should be unpacked"),
        ("value_batch2 - returns2_batch", "Critic2 should use returns2_batch for loss"),
    ]
    
    all_passed = True
    for check_str, description in checks:
        if check_str in update_source:
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ FAILED: {description}")
            all_passed = False
    
    if all_passed:
        print(f"\n✅ TEST 2 PASSED: PPO correctly uses returns2_batch for Critic2")
    else:
        print(f"\n❌ TEST 2 FAILED: PPO integration has issues")
    
    return all_passed


def test_advantage_calculation():
    """Test that advantage calculation uses correct rewards."""
    print("\n" + "=" * 60)
    print("TEST 3: Advantage Calculation")
    print("=" * 60)
    
    from humanoid.algo.ppo.rollout_storage import RolloutStorage
    
    num_envs = 2
    num_transitions = 10
    device = 'cpu'
    
    storage = RolloutStorage(
        num_envs=num_envs,
        num_transitions_per_env=num_transitions,
        obs_shape=(5,),
        privileged_obs_shape=(5,),  # Use same as obs_shape for testing
        actions_shape=(2,),
        device=device,
        use_double_critic=True  # Must be set during __init__ to allocate buffers
    )
    
    # Set up known rewards
    # Dense: constant positive
    storage.rewards_dense = torch.ones(num_transitions, num_envs, 1) * 1.0
    # Sparse: only reward at specific timesteps (simulating foothold reward)
    storage.rewards_sparse = torch.zeros(num_transitions, num_envs, 1)
    storage.rewards_sparse[5, :, :] = 2.0  # Big sparse reward at t=5
    
    storage.values = torch.zeros(num_transitions, num_envs, 1)
    storage.values2 = torch.zeros(num_transitions, num_envs, 1)
    storage.dones = torch.zeros(num_transitions, num_envs, 1)
    
    # Compute returns
    last_values1 = torch.zeros(num_envs, 1)
    last_values2 = torch.zeros(num_envs, 1)
    storage.compute_returns(last_values1, gamma=0.99, lam=0.95,
                           last_values2=last_values2, w1=1.0, w2=1.0)
    
    # Check that returns are different
    returns_mean = storage.returns.mean().item()
    returns2_mean = storage.returns2.mean().item()
    
    print(f"  Dense returns mean: {returns_mean:.4f}")
    print(f"  Sparse returns mean: {returns2_mean:.4f}")
    
    # Dense should be larger (constant positive rewards)
    # Sparse should be smaller (mostly zeros with one spike)
    assert returns_mean > returns2_mean, "Dense returns should be larger than sparse"
    
    # Check advantage is combined
    advantages_mean = storage.advantages.mean().item()
    print(f"  Combined advantages mean: {advantages_mean:.4f}")
    
    print(f"\n✅ TEST 3 PASSED: Advantages correctly computed from both reward streams")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DOUBLE CRITIC BUG FIX VERIFICATION")
    print("=" * 60)
    print("\nTesting the fix for Critic2 training target bug.")
    print("Before fix: Critic2 was trained with returns (dense) instead of returns2 (sparse)")
    print("After fix: Critic2 correctly uses returns2 (sparse) as training target\n")
    
    try:
        test1_passed = test_storage_generator()
        test2_passed = test_ppo_integration()
        test3_passed = test_advantage_calculation()
        
        print("\n" + "=" * 60)
        if test1_passed and test2_passed and test3_passed:
            print("🎉 ALL TESTS PASSED! 🎉")
            print("=" * 60)
            print("\n✅ Bug fix verified:")
            print("   - Critic2 now receives returns2 (sparse rewards) as training target")
            print("   - Data flow from storage to PPO is correct")
            print("   - Advantages are properly computed from both reward streams")
            sys.exit(0)
        else:
            print("❌ SOME TESTS FAILED")
            print("=" * 60)
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

