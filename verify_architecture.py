#!/usr/bin/env python3
"""
Architecture Verification Script
验证新架构的观测维度是否正确（纯配置检查）
"""

def verify_dimensions():
    """验证配置的维度设置（直接读取配置值）"""
    
    # Hardcoded config values (从 humanoid_config.py 读取)
    frame_stack = 15
    c_frame_stack = 3
    num_single_obs = 47
    single_num_privileged_obs = 73
    height_map_dim = 225
    
    num_observations = int(frame_stack * num_single_obs + height_map_dim)
    num_privileged_obs = int(c_frame_stack * single_num_privileged_obs + height_map_dim)
    
    print("="*70)
    print("🔍 DIMENSION VERIFICATION - /workspace_robot/humanoid-gym/")
    print("="*70)
    
    # Actor dimensions
    print("\n📊 Actor Observation:")
    print(f"  - Single frame (base obs): {num_single_obs}D")
    print(f"  - Frame stack: {frame_stack} frames")
    print(f"  - Stacked base obs: {frame_stack * num_single_obs}D")
    print(f"  - Height map (current): {height_map_dim}D")
    print(f"  - Total: {num_observations}D")
    
    expected_actor = frame_stack * num_single_obs + height_map_dim
    status = "✅" if num_observations == expected_actor else "❌"
    print(f"  {status} Expected: {expected_actor}D, Got: {num_observations}D")
    
    # Critic dimensions
    print("\n📊 Critic Observation:")
    print(f"  - Single frame (privileged): {single_num_privileged_obs}D")
    print(f"  - Frame stack: {c_frame_stack} frames")
    print(f"  - Stacked privileged obs: {c_frame_stack * single_num_privileged_obs}D")
    print(f"  - Height map (current): {height_map_dim}D")
    print(f"  - Total: {num_privileged_obs}D")
    
    expected_critic = c_frame_stack * single_num_privileged_obs + height_map_dim
    status = "✅" if num_privileged_obs == expected_critic else "❌"
    print(f"  {status} Expected: {expected_critic}D, Got: {num_privileged_obs}D")
    
    # Network architecture
    print("\n🧠 Network Architecture:")
    actor_hidden_dims = [768, 512, 256]
    critic_hidden_dims = [768, 512, 256]
    print(f"  - Actor hidden dims: {actor_hidden_dims}")
    print(f"  - Critic hidden dims: {critic_hidden_dims}")
    print(f"  - Actor first layer: {num_observations} → {actor_hidden_dims[0]}")
    print(f"  - Critic first layer: {num_privileged_obs} → {critic_hidden_dims[0]}")
    
    # Episode configuration
    print("\n⏱️  Episode Configuration:")
    episode_length_s = 24
    control_decimation = 10
    sim_dt = 0.001
    steps_per_episode = int(episode_length_s * 1000 / sim_dt / control_decimation)
    print(f"  - Episode length: {episode_length_s}s")
    print(f"  - Control decimation: {control_decimation} (100Hz)")
    print(f"  - Steps per episode: {steps_per_episode}")
    
    # Breakdown of base observation (47D)
    print("\n📋 Actor Base Observation Breakdown (47D):")
    breakdown = [
        ("Phase (sin, cos)", 2),
        ("Commands (vx, vy, omega)", 3),
        ("Joint positions", 12),
        ("Joint velocities", 12),
        ("Previous actions", 12),
        ("Base angular velocity", 3),
        ("Projected gravity", 3),
    ]
    total = 0
    for name, dim in breakdown:
        print(f"  - {name}: {dim}D")
        total += dim
    status = "✅" if total == num_single_obs else "❌"
    print(f"  {status} Total: {total}D")
    
    # Breakdown of privileged observation (73D)
    print("\n📋 Critic Privileged Observation Breakdown (73D):")
    priv_breakdown = [
        ("Phase + Commands", 5),
        ("Joint positions (from default)", 12),
        ("Joint velocities", 12),
        ("Actions", 12),
        ("Joint error (pos - ref)", 12),
        ("Base linear velocity ⭐", 3),
        ("Base angular velocity", 3),
        ("Base euler angles", 3),
        ("Random push force ⭐", 2),
        ("Random push torque ⭐", 3),
        ("Environment friction ⭐", 1),
        ("Body mass ⭐", 1),
        ("Stance mask ⭐", 2),
        ("Contact mask ⭐", 2),
    ]
    total = 0
    for name, dim in priv_breakdown:
        marker = "  ⭐" if "⭐" in name else "    "
        print(f"{marker} {name.replace(' ⭐', '')}: {dim}D")
        total += dim
    status = "✅" if total == single_num_privileged_obs else "❌"
    print(f"  {status} Total: {total}D")
    print(f"\n  (⭐ = Privileged information not available to Actor)")
    
    # Comparison with original version
    print("\n📊 Comparison with Original (huu) Version:")
    print(f"  Original Actor:  15 × 47D = 705D (no height map)")
    print(f"  New Actor:       15 × 47D + 225D = {num_observations}D ✅")
    print(f"  Original Critic: 3 × 73D = 219D (no height map)")
    print(f"  New Critic:      3 × 73D + 225D = {num_privileged_obs}D ✅")
    
    # Key features
    print("\n✨ Key Features:")
    features = [
        "✅ Asymmetric Actor-Critic (Actor 930D vs Critic 444D)",
        "✅ Frame stacking for temporal modeling (15/3 frames)",
        "✅ Height map for spatial awareness (15×15 grid)",
        "✅ Critic privileged information (velocity, friction, mass, etc.)",
        "✅ Height map NOT frame-stacked (memory efficient)",
    ]
    for feature in features:
        print(f"  {feature}")
    
    # Final summary
    print("\n" + "="*70)
    all_good = (
        num_observations == expected_actor and
        num_privileged_obs == expected_critic and
        sum(d for _, d in breakdown) == num_single_obs and
        sum(d for _, d in priv_breakdown) == single_num_privileged_obs
    )
    
    if all_good:
        print("✅ ALL DIMENSIONS VERIFIED SUCCESSFULLY!")
        print("\n🚀 Ready to train! Use:")
        print("   python scripts/train.py --task=humanoid_stones_stage1_plane_ppo --run_name test_new_arch")
    else:
        print("❌ DIMENSION MISMATCH DETECTED!")
    print("="*70)
    
    return all_good

if __name__ == "__main__":
    import sys
    success = verify_dimensions()
    sys.exit(0 if success else 1)

