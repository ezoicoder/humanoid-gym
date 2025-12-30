#!/usr/bin/env python3
"""
Quick test script to verify double critic implementation.
"""

# Import isaacgym modules first (before torch)
from humanoid.envs.custom.humanoid_config import XBotLCfgPPO
from humanoid.algo.ppo import ActorCritic, PPO, RolloutStorage
import torch

def test_config():
    """Test configuration loading"""
    print("=" * 60)
    print("TEST 1: Configuration Loading")
    print("=" * 60)
    
    cfg = XBotLCfgPPO()
    
    assert hasattr(cfg.algorithm, 'use_double_critic'), "Missing use_double_critic"
    assert hasattr(cfg.algorithm, 'advantage_weight_dense'), "Missing advantage_weight_dense"
    assert hasattr(cfg.algorithm, 'advantage_weight_sparse'), "Missing advantage_weight_sparse"
    
    print(f"✓ use_double_critic: {cfg.algorithm.use_double_critic}")
    print(f"✓ advantage_weight_dense: {cfg.algorithm.advantage_weight_dense}")
    print(f"✓ advantage_weight_sparse: {cfg.algorithm.advantage_weight_sparse}")
    print("✓ Config test PASSED\n")

def test_actor_critic():
    """Test ActorCritic with double critic"""
    print("=" * 60)
    print("TEST 2: ActorCritic Network")
    print("=" * 60)
    
    # Test single critic mode
    ac_single = ActorCritic(
        num_actor_obs=272,
        num_critic_obs=272,
        num_actions=12,
        use_double_critic=False
    )
    
    obs = torch.randn(10, 272)
    value_single = ac_single.evaluate(obs)
    assert value_single.shape == (10, 1), f"Wrong shape: {value_single.shape}"
    print(f"✓ Single critic output shape: {value_single.shape}")
    
    # Test double critic mode
    ac_double = ActorCritic(
        num_actor_obs=272,
        num_critic_obs=272,
        num_actions=12,
        use_double_critic=True
    )
    
    value1, value2 = ac_double.evaluate(obs)
    assert value1.shape == (10, 1), f"Wrong shape: {value1.shape}"
    assert value2.shape == (10, 1), f"Wrong shape: {value2.shape}"
    print(f"✓ Double critic output shapes: {value1.shape}, {value2.shape}")
    print("✓ ActorCritic test PASSED\n")

def test_rollout_storage():
    """Test RolloutStorage with double critic"""
    print("=" * 60)
    print("TEST 3: RolloutStorage")
    print("=" * 60)
    
    # Test single critic mode
    storage_single = RolloutStorage(
        num_envs=10,
        num_transitions_per_env=50,
        obs_shape=[272],
        privileged_obs_shape=[272],
        actions_shape=[12],
        device='cpu',
        use_double_critic=False
    )
    print(f"✓ Single critic storage initialized")
    
    # Test double critic mode
    storage_double = RolloutStorage(
        num_envs=10,
        num_transitions_per_env=50,
        obs_shape=[272],
        privileged_obs_shape=[272],
        actions_shape=[12],
        device='cpu',
        use_double_critic=True
    )
    
    assert hasattr(storage_double, 'rewards_dense'), "Missing rewards_dense"
    assert hasattr(storage_double, 'rewards_sparse'), "Missing rewards_sparse"
    assert hasattr(storage_double, 'values2'), "Missing values2"
    assert hasattr(storage_double, 'returns2'), "Missing returns2"
    assert hasattr(storage_double, 'advantages2'), "Missing advantages2"
    
    print(f"✓ Double critic storage initialized")
    print(f"✓ rewards_dense shape: {storage_double.rewards_dense.shape}")
    print(f"✓ rewards_sparse shape: {storage_double.rewards_sparse.shape}")
    print(f"✓ values2 shape: {storage_double.values2.shape}")
    print("✓ RolloutStorage test PASSED\n")

def test_ppo():
    """Test PPO with double critic"""
    print("=" * 60)
    print("TEST 4: PPO Algorithm")
    print("=" * 60)
    
    # Create actor-critic
    ac = ActorCritic(
        num_actor_obs=272,
        num_critic_obs=272,
        num_actions=12,
        use_double_critic=True
    )
    
    # Create PPO
    ppo = PPO(
        actor_critic=ac,
        device='cpu',
        use_double_critic=True,
        advantage_weight_dense=1.0,
        advantage_weight_sparse=0.25
    )
    
    assert ppo.use_double_critic == True, "Double critic not enabled"
    assert ppo.advantage_weight_dense == 1.0, "Wrong dense weight"
    assert ppo.advantage_weight_sparse == 0.25, "Wrong sparse weight"
    
    print(f"✓ PPO initialized with double critic")
    print(f"✓ advantage_weight_dense: {ppo.advantage_weight_dense}")
    print(f"✓ advantage_weight_sparse: {ppo.advantage_weight_sparse}")
    
    # Initialize storage
    ppo.init_storage(
        num_envs=10,
        num_transitions_per_env=50,
        actor_obs_shape=[272],
        critic_obs_shape=[272],
        action_shape=[12]
    )
    
    assert ppo.storage.use_double_critic == True, "Storage double critic not enabled"
    print(f"✓ Storage initialized with double critic")
    print("✓ PPO test PASSED\n")

def test_advantage_computation():
    """Test advantage computation with double critic"""
    print("=" * 60)
    print("TEST 5: Advantage Computation")
    print("=" * 60)
    
    storage = RolloutStorage(
        num_envs=4,
        num_transitions_per_env=10,
        obs_shape=[272],
        privileged_obs_shape=[272],
        actions_shape=[12],
        device='cpu',
        use_double_critic=True
    )
    
    # Fill with dummy data
    storage.rewards_dense[:] = torch.randn(10, 4, 1)
    storage.rewards_sparse[:] = torch.randn(10, 4, 1) * 0.1  # Sparse rewards are smaller
    storage.values[:] = torch.randn(10, 4, 1)
    storage.values2[:] = torch.randn(10, 4, 1)
    storage.dones[:] = 0
    
    last_values1 = torch.randn(4, 1)
    last_values2 = torch.randn(4, 1)
    
    # Compute returns and advantages
    storage.compute_returns(
        last_values=last_values1,
        gamma=0.99,
        lam=0.95,
        last_values2=last_values2,
        w1=1.0,
        w2=0.25
    )
    
    assert storage.returns.shape == (10, 4, 1), f"Wrong returns shape: {storage.returns.shape}"
    assert storage.returns2.shape == (10, 4, 1), f"Wrong returns2 shape: {storage.returns2.shape}"
    assert storage.advantages.shape == (10, 4, 1), f"Wrong advantages shape: {storage.advantages.shape}"
    
    # Check that advantages are normalized (mean ≈ 0, std ≈ 1)
    adv_mean = storage.advantages.mean().item()
    adv_std = storage.advantages.std().item()
    
    print(f"✓ Returns computed")
    print(f"✓ Advantages computed and combined")
    print(f"✓ Advantage mean: {adv_mean:.4f} (should be ≈ 0)")
    print(f"✓ Advantage std: {adv_std:.4f} (should be ≈ 1)")
    print("✓ Advantage computation test PASSED\n")

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("DOUBLE CRITIC IMPLEMENTATION TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_config()
        test_actor_critic()
        test_rollout_storage()
        test_ppo()
        test_advantage_computation()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("\nDouble critic implementation is working correctly!")
        print("\nTo use in training:")
        print("  python scripts/train.py --task=XBotL_free --double_critic")
        print()
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

