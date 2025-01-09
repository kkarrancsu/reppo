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
The simulation models user staking behavior through:

- **Staking Rate**: Follows a Poisson process with rate λ(t) influenced by:
  - Current returns (r(t))
  - Market conditions (m(t))
  - Recent performance (p(t))
  - λ(t) = base_rate * (1 + r(t)) * m(t) * p(t)

- **Lock Duration Choice**: Probabilistic selection weighted by expected returns:
  - P(D) ∝ exp(r * (D/D_max)^γ)
  - Longer durations more likely when returns are higher
  - γ parameter controls non-linearity of this relationship

### 3. Pod Performance
Each pod generates fees based on:

- **Base Fee Generation**:
  - dF_p = μ_p * dt + σ_p * dW_t
  - μ_p = fee_drift * votes * market_rate
  - σ_p = fee_volatility * sqrt(votes) * market_rate

- **Vote Distribution**:
  - Updated each epoch based on performance
  - Incorporates both performance chasing (α) and diversification (δ)
  - V_p ~ Dirichlet(α * F_p/F_total + δ)

### 4. FCU Mechanics
FCUs represent claims on future pod fees:

- **Generation**: Based on vote allocation and fee generation
- **Activation**: After pod-specific delay period
- **Duration**: Active for τ epochs once activated
- **Fee Distribution**: Pro-rata share of pod's distributable fees

## Simulation Steps

Each epoch, the simulation:

1. **Process Events**
   - Lock expirations
   - FCU activations
   - Emission distributions
   - Performance checks

2. **Market Update**
   - Updates base fee rate with growth and volatility
   - Affects pod fee generation capacity

3. **Staking**
   - Generates new staking events
   - Assigns lock durations
   - Updates veToken power

4. **Fee Generation**
   - Each pod generates fees
   - Distributes fees to FCU holders
   - Updates performance metrics

5. **Vote Reallocation**
   - Updates vote distribution based on performance
   - Applies diversification incentives

6. **FCU Generation**
   - Issues new FCUs based on vote allocation
   - Schedules activation events

7. **Metrics Update**
   - Calculates system stability metrics
   - Updates historical records
   - Tracks pod-specific performance

## Key Parameters

### Protocol Parameters
- **γ (Gamma)**: Controls veToken power scaling with lock duration
- **α (Alpha)**: Weight of performance in vote allocation
- **δ (Delta)**: Base weight for vote diversification
- **ω (Omega)**: FCU generation rate

### Market Parameters
- **Base Fee Rate**: Underlying rate for fee generation
- **Growth Rate**: Market growth trend
- **Volatility**: Market uncertainty

### Pod Parameters
- **Fee Drift**: Base rate of fee generation
- **Fee Split**: Portion of fees distributed to FCU holders
- **FCU Delay**: Time before FCUs become active

### System Parameters
- **Lock Durations**: Min and max allowed lock periods
- **Initial Supply**: Starting token supply
- **Emission Schedule**: Token emission rate over time
""")


# """
# ## Success Metrics

# The simulation tracks several key metrics to evaluate system health:

# 1. **Economic Metrics**
#    - Fee generation efficiency
#    - Emission vs fee balance
#    - TVL growth

# 2. **Participation Metrics**
#    - Active positions
#    - Average lock duration
#    - Vote distribution entropy

# 3. **Pod Performance**
#    - Individual pod fee generation
#    - FCU efficiency
#    - Vote share stability

# 4. **Market Indicators**
#    - Total fees vs emissions
#    - Vesting progress
#    - Market rate evolution
# """