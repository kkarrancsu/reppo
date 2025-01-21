import streamlit as st

st.set_page_config(
    page_title="Reppo VeTokenomics Simulation",
    layout="wide",
)

st.markdown("""
# VeTokenomics Simulation Overview

This simulation models a veTokenomics system where users can lock tokens to gain voting power and earn rewards from different "Pods" - units of intelligence in the Reppo network. The simulation captures the complex interactions between stakers, pods, and market dynamics.

## Core Components

### 1. Token System
- **Base Tokens**: The underlying token that users can lock
- **veTokens**: Voting power tokens received for locking, weighted by lock duration
- **FCUs (Fee Claim Units)**: Claims on future pod fees, earned based on vote allocation

### 2. Staking Dynamics

For the stochastic simualtion, staking behavior is modeled as a Poisson process with rate λ(t) influenced by:
- **Staking Rate**: 
  - Current returns (r(t))
  - Market conditions (m(t))
  - Recent performance (p(t))
  - λ(t) = base_rate * (1 + r(t)) * m(t) * p(t)

- **Lock Duration Choice**: Probabilistic selection weighted by expected returns:
  - P(D) ∝ exp(r * (D/D_max)^γ)
  - Longer durations more likely when returns are higher
  - γ parameter controls non-linearity of this relationship

For the deterministic simulation, staking rate is a constant that is set by the user.

### 3. Pod Performance
For the stochastic simulation, pod performance is modeled as a geometric Brownian motion with drift and volatility:
- **Base Fee Generation**:
  - dF_p = μ_p * dt + σ_p * dW_t
  - μ_p = fee_drift * votes * market_rate
  - σ_p = fee_volatility * sqrt(votes) * market_rate

- **Vote Distribution**:
  - Updated each epoch based on performance
  - Incorporates both performance chasing (α) and diversification (δ)
  - V_p ~ Dirichlet(α * F_p/F_total + δ)

For the deterministic simulation, pod performance is not computed directy, but the user configures
a constant growth rate of fees that the pod generates.

### 4. FCU Mechanics
FCUs represent claims on future pod fees:

- **Generation**: Based on vote allocation and fee generation
- **Activation**: After pod-specific delay period
- **Duration**: Active for τ epochs once activated
- **Fee Distribution**: Pro-rata share of pod's distributable fees

NOTE: the simulation currently expires FCUs, but we have decided to remove this feature.  This is an update that needs to be made, but will simplify the simulation.

## Simulation Description

### Data Structures
The following data structures are useful to help understand the simulation:

LockState (tracks a lock position - i.e. stake)
            
    - amount: float
    - duration: float
    - start_epoch: float
    - fcu_claims: float
    - earned_fees: float
            
PodState (tracks a Pod)
            
    - fees: float
    - votes: float
    - fcus: float
    - fee_drift: Base rate of fee generation
    - emissions: emissions directed to the pod
    - fee_split: Portion of fees distributed to FCU holders
    - fcu_generation_rate: Rate at which FCUs are generated
            
MarketState (tracks the market)
            
    - base_fee_rate: Underlying rate for fee generation
    - growth_rate: Market growth trend
    - volatility: Market uncertainty
    - total_fees: float
    - total_fcus: float
    - total_emissions: float
    - peak_votes: float
    - min_votes: float
    - cumulative_vote_share: float
    - vote_samples: int
            
SimulationParams (simulation configuration)
            
    - gamma: Controls veToken power scaling with lock duration
    - alpha: Weight of performance in vote allocation
    - delta: Base weight for vote diversification
    - omega: FCU generation rate
    - base_rate: Base staking rate
    - lock_duration_min: Minimum lock duration
    - lock_duration_max: Maximum lock duration
    - initial_supply: Starting token supply
    - emission_schedule: Token emission rate over time
            
### Simulation Flow
The simulation runs in a loop and is an event based simulator.  Events are things such as token emissions, FCU activations, lock expirations, etc. The events
trigger at a given epoch, and the simulation simply executes the events in time-order.  Events can trigger new events, thereby continuing the simulation for
a set period of of time.
            
For each epoch, the following steps are taken:

1. **Process Events**
   - Emission distributions
   - FCU activations
   - Lock expirations (i.e. veReppo ==> Reppo)

2. **Market Update**
   - Updates base fee rate with growth and volatility (in stochastic simulation)

3. **Staking**
   - Generates new staking events (in both stochastic and deterministic simulations)
   - Assigns lock durations (in stochastic simulation)
   - Updates veToken power

4. **Fee Generation**
   - Each pod generates fees
   - Distributes fees to FCU holders
   - Updates performance metrics

5. **Vote Reallocation**
   - Updates vote distribution based on performance
   - Applies diversification incentives
   - TODO: need to add in a parameter on how often this is happening, every N epochs rather than every epoch (current behavior)

6. **FCU Generation**
   - Issues new FCUs based on vote allocation
   - Schedules activation events

7. **Metrics Update**
   - Calculates system stability metrics
   - Updates historical records
   - Tracks pod-specific performance
""")