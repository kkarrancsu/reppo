from typing import List
import pytest
from reppo.vesting import VestingManager

def test_overlapping_linear_vesting():
    # Setup vesting manager with 14 epoch duration
    manager = VestingManager(vesting_duration=14)
    
    # Add 100 tokens starting at epoch 0
    manager.add_emission(100.0, start_epoch=0)
    
    # Add another 100 tokens starting at epoch 7
    manager.add_emission(100.0, start_epoch=7)

    # Expected values for each epoch
    # Single schedule until epoch 6: 100/14 ≈ 7.14 per epoch
    # Overlapping schedules from epoch 7-13: (100/14 + 100/14) ≈ 14.28 per epoch
    # Single schedule from epoch 14-20: 100/14 ≈ 7.14 per epoch
    
    # Test initial schedule only
    assert pytest.approx(manager.tokens_to_release(0)) == 100/14
    assert pytest.approx(manager.tokens_to_release(6)) == 100/14
    
    # Test overlapping period
    assert pytest.approx(manager.tokens_to_release(7)) == (100/14) * 2
    assert pytest.approx(manager.tokens_to_release(13)) == (100/14) * 2
    
    # Test second schedule only
    assert pytest.approx(manager.tokens_to_release(14)) == 100/14
    assert pytest.approx(manager.tokens_to_release(20)) == 100/14
    
    # Test after all vesting completed
    assert manager.tokens_to_release(21) == 0

    # Verify total remaining tokens at key points
    assert pytest.approx(manager.remaining_tokens(0)) == 200  # All tokens remaining
    assert pytest.approx(manager.remaining_tokens(7)) == 200 - (7 * 100/14)  # After first week
    assert pytest.approx(manager.remaining_tokens(14)) == 100 - (7 * 100/14)  # After first schedule
    assert pytest.approx(manager.remaining_tokens(21)) == 0  # All vested

def run_tests():
    test_overlapping_linear_vesting()
    print("All tests passed!")

if __name__ == "__main__":
    run_tests()