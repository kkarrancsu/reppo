import numpy as np
from reppo.vetoken import \
    VeTokenomicsSimulation, \
    SimulationParams, \
    MarketState, \
    VestingSchedule, \
    SystemState

def custom_emission_schedule(state: SystemState) -> float:
    """
    Custom emission schedule that considers current network state
    """
    base = 1000  # Base emission per epoch
    decay = 0.95  # Decay factor
    utilization = state.locked_tokens / state.total_supply if state.total_supply > 0 else 0
    
    # Emission reduces over time and increases with utilization
    emission = base * (decay ** state.epoch) * (1 + utilization)
    return emission

vesting = VestingSchedule(
    initial_amount=1_000_000,  # 1M tokens
    cliff_epoch=26,            # 6 month cliff
    vesting_duration=104,      # 2 year vesting
    release_frequency=13       # Quarterly releases
)

params = SimulationParams(
    gamma=2.0,
    alpha=0.5,
    delta=0.2,
    omega=0.1,
    fee_volatility=0.1,
    base_stake_rate=0.1,
    base_fee_drift=0.05,
    max_lock_duration=52,
    min_lock_duration=4,
    initial_pods=["pod1", "pod2"],
    initial_token_supply=1_000_000,
    epochs=120,
    market=MarketState(
        base_fee_rate=0.1,
        growth_rate=0.05,
        volatility=0.2
    ),
    vesting=vesting
)

sim = VeTokenomicsSimulation(params)

# Create some locks
sim.create_lock(amount=10000, duration=26)  # 6 month lock
sim.create_lock(amount=20000, duration=52)  # 1 year lock

# Run simulation
states = sim.run()
print(states)