python -c "import time; print(time.time()); time.sleep(2600); print(time.time())"
script -c "python scripts/train.py --task=humanoid_stones_stage1_plane_ppo --run_name denser-reward-dense-heights --double_critic --headless --platform_width=3.0 --platform_length=3.0" -f ../logs/stage1_denser.txt
