from typing import Dict, List, Optional, NamedTuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from numpy.random import Generator, PCG64
from vesting import VestingSchedule, VestingManager
from collections import defaultdict

class EventType(Enum):
    LOCK_EXPIRY = "lock_expiry"
    FCU_ACTIVATION = "fcu_activation" 
    EMISSION_DIST = "emission_dist"
    PERF_CHECK = "perf_check"
    FCU_CLAIM = "fcu_claim"
    FCU_EXPIRY = "fcu_expiry"
@dataclass
class Event:
    type: EventType
    epoch: int
    data: dict

@dataclass
class MarketState:
    base_fee_rate: float  # Base rate for fee generation
    growth_rate: float    # Expected market growth rate
    volatility: float     # Market volatility

    def __post_init__(self):
        self.initial_base_fee_rate = self.base_fee_rate
    
    def apply_market_shock(self, shock: float) -> None:
        self.base_fee_rate *= (1 + shock)
        if self.base_fee_rate < 0:
            self.base_fee_rate = 0
            
    def apply_growth(self) -> None:
        self.base_fee_rate *= (1 + self.growth_rate)

    def get_market_factor(self) -> float:
        return self.base_fee_rate / self.initial_base_fee_rate if self.initial_base_fee_rate > 0 else 1.0

class SystemState:
    def __init__(self, params):
        self.epoch = 0
        self.total_supply = params.initial_token_supply
        self.locked_tokens = 0.0
        self.ve_tokens = 0.0
        self.lock_positions: List[LockPosition] = []

        initial_vote_share = 1.0 / len(params.initial_pods)
        
        self.pods: Dict[str, PodState] = {
            pod: PodState(0.0, 1.0 / len(params.initial_pods), 0.0, params.base_fee_drift)
            for pod in params.initial_pods
        }

        self.market_state = params.market
        self.metrics = {
            'avg_lock_duration': 0.0,
            'active_positions': 0,
            'vote_entropy': 0.0,
            
            'emissions': {
                'current': 0.0,  # Emissions in current epoch
                'total': 0.0,    # Cumulative emissions
                'vested': 0.0,   # Only used with VestingManager
                'unvested': 0.0  # Only used with VestingManager
            },
            'pods': {}
        }

    def update_metrics(self) -> None:
        active_positions = [p for p in self.lock_positions if not p.is_expired(self.epoch)]
        if active_positions:
            self.metrics['avg_lock_duration'] = float(np.mean([p.duration for p in active_positions]))
            self.metrics['active_positions'] = len(active_positions)
        
        pod_metrics = {}
        for name, pod in self.pods.items():
            active_claims = getattr(pod, 'active_claims', [])
            serializable_claims = len(active_claims)

            pod_metrics[name] = {
                'emissions': {
                    'current': pod.emissions,
                    'total': pod.total_emissions
                },
                'fees': {
                    'current': pod.fees,
                    'total': pod.total_fees,
                    'distributed': pod.fees * pod.fee_split
                },
                'fcus': {
                    'current': pod.fcus,
                    'total': pod.total_fcus,
                    'active': serializable_claims 
                },
                'votes': {
                    'current': pod.votes,
                    'peak': pod.peak_votes,
                    'min': pod.min_votes,
                    'avg': pod.avg_vote_share
                },
                'efficiency': {
                    'fee_rate': float(pod.fees / max(pod.votes, 0.0001)),
                    'fcu_rate': float(pod.fcus / max(pod.fees, 0.0001))
                }
            }
        
        self.metrics['pods'] = pod_metrics
        
        votes = np.array([pod.votes for pod in self.pods.values()])
        votes = votes[votes > 0]
        if len(votes) > 0:
            self.metrics['vote_entropy'] = float(-np.sum(votes * np.log(votes)))
        
    def get_total_fees(self) -> float:
        return sum(pod.fees for pod in self.pods.values())
    
    def get_vote_distribution(self) -> Dict[str, float]:
        return {name: pod.votes for name, pod in self.pods.items()}
    
    def get_locked_tokens(self) -> float:
        return self.locked_tokens

@dataclass
class PodState:
    fees: float
    votes: float
    fcus: float
    fee_drift: float
    emissions: float = 0.0  # emissions received by pod
    fee_split: float = 0.7  # αp - fraction of fees distributed to FCU holders
    fcu_generation_rate: float = 1.0  # controls FCU generation 
    
    # Historical metrics
    total_fees: float = 0.0
    total_fcus: float = 0.0
    total_emissions: float = 0.0
    peak_votes: float = 0.0
    min_votes: float = 1.0
    cumulative_vote_share: float = 0.0
    vote_samples: int = 0
    
    def update_metrics(self) -> None:
        self.total_fees += self.fees
        self.total_fcus += self.fcus
        self.total_emissions += self.emissions
        self.peak_votes = max(self.peak_votes, self.votes)
        self.min_votes = min(self.min_votes, self.votes)
        self.cumulative_vote_share += self.votes
        self.vote_samples += 1
        
    @property
    def avg_vote_share(self) -> float:
        return self.cumulative_vote_share / self.vote_samples if self.vote_samples > 0 else 0
    
@dataclass
class SimulationParams:
    gamma: float  
    alpha: float  
    delta: float  
    omega: float  
    fee_volatility: float
    base_stake_rate: float
    base_fee_drift: float
    max_lock_duration: int
    min_lock_duration: int  # Added minimum lock duration
    initial_pods: List[str]
    initial_token_supply: float
    epochs: int
    market: MarketState

    fcu_duration: int  # τ in the paper - how long FCUs last
    fcu_delay: Dict[str, int]  # δp for each pod - delay before FCUs activate
    pod_fcu_rates: Dict[str, float]  # FCU generation rates per pod

    # vesting: Optional[VestingSchedule] = None
    emission_schedule: Optional[Callable[[SystemState], float]] = None
    emission_vesting_duration: Optional[int] = None  # None means instant vesting
    
    # # features to be included soon...
    # min_viable_fees: float 
    # max_vote_concentration: float
    # min_fee_coverage: float
    # pod_correlation_matrix: np.ndarray = np.eye(len(initial_pods))

@dataclass
class FCUClaim:
    staker_index: int  # Index of the LockPosition/staker
    pod_name: str     # Pod the claim is for
    amount: float     # Amount of FCUs (based on their effective votes * ω)
    acquisition_epoch: int  # When earned
    activation_epoch: int   # When claim becomes active (acquisition + δp)
    expiry_epoch: int      # When claim expires (activation + τ)

@dataclass
class LockPosition:
    amount: float
    duration: int
    start_epoch: int
    fcu_claims: Dict[str, List[FCUClaim]] = field(default_factory=lambda: defaultdict(list))
    earned_fees: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    
    def is_expired(self, current_epoch: int) -> bool:
        return current_epoch >= self.start_epoch + self.duration
        
    def ve_power(self, params: SimulationParams) -> float:
        duration_weight = (self.duration / params.max_lock_duration) ** params.gamma
        return self.amount * duration_weight
        
    def extend_duration(self, new_duration: int) -> None:
        if new_duration <= self.duration:
            raise ValueError("New duration must be greater than current duration")
        self.duration = new_duration

class VeTokenomicsSimulation:
    def __init__(self, params: SimulationParams):
        self.params = params
        self.state = SystemState(params)
        self.events: List[Event] = []
        self.rng = np.random.Generator(PCG64())
        self.history: List[dict] = []
        if params.emission_vesting_duration:
            self.vesting_manager = VestingManager(params.emission_vesting_duration)
        else:
            self.vesting_manager = None
        self._init_events()

    def _init_events(self) -> None:
        self.events.append(Event(
            type=EventType.EMISSION_DIST,
            epoch=1,
            data={}
        ))

    def run(self, epochs: Optional[int] = None) -> List[SystemState]:
        max_epochs = epochs or self.params.epochs
        states = [self.state]
        
        for _ in range(max_epochs):
            state = self.step()
            states.append(state)
            
        return states
    
    def step(self) -> SystemState:
        # Zero out pod emissions from previous epoch
        for pod in self.state.pods.values():
            pod.emissions = 0.0

        self._process_events()
        self._simulate_market()
        self._simulate_staking()
        self._process_fees()
        self._update_votes()
        self._generate_fcus()
        self._update_ve_tokens()
        
        self.state.update_metrics()
        self._record_state()

        print(f"\nEnd of Epoch {self.state.epoch} Summary:")
        for name, pod in self.state.pods.items():
            print(f"{name}:")
            print(f"  Fees: {pod.fees:.4f}")
            print(f"  Total Fees: {pod.total_fees:.4f}")
            print(f"  Votes: {pod.votes:.4f}")
            print(f"  FCUs: {pod.fcus:.4f}")
            print(f"  Emissions: {pod.emissions:.4f}")
            print(f"  Total Emissions: {pod.total_emissions:.4f}")
        self.state.epoch += 1
        return self.state
    
    def _process_events(self) -> None:
        current_events = [e for e in self.events if e.epoch == self.state.epoch]
        for event in current_events:
            if event.type == EventType.EMISSION_DIST:
                self._process_emission()
                self.events.append(Event(
                    type=EventType.EMISSION_DIST,
                    epoch=self.state.epoch + 1,
                    data={}
                ))
                
            elif event.type == EventType.PERF_CHECK:
                self._check_pod_performance()
                self.events.append(Event(
                    type=EventType.PERF_CHECK,
                    epoch=self.state.epoch + 1,
                    data={}
                ))
                
            elif event.type == EventType.FCU_ACTIVATION:
                self._process_fcu_activation(event.data)
                
            elif event.type == EventType.LOCK_EXPIRY:
                self._process_lock_expiry(event.data["position_index"])

            # TODO: can process FCU expiry if we want here ...
        self.events = [e for e in self.events if e.epoch != self.state.epoch]
    
    def _simulate_market(self) -> None:
        growth = self.state.market_state.growth_rate
        vol = self.state.market_state.volatility
        shock = self.rng.normal(0, vol)
        
        self.state.market_state.base_fee_rate *= np.exp(growth + shock)
    
    def _simulate_staking(self) -> None:
        base_rate = self.params.base_stake_rate  # Base staking rate
        
        # Calculate factors that influence staking rate
        returns = self.state.get_total_fees() / max(self.state.locked_tokens, 1)
        market_factor = self.state.market_state.get_market_factor()
        performance = sum(pod.avg_vote_share * pod.total_fees for pod in self.state.pods.values())
        
        # Calculate lambda for Poisson process
        lambda_t = base_rate * (1 + returns) * market_factor * (1 + performance)
        
        # Generate number of new stakes this epoch
        num_stakes = self.rng.poisson(lambda_t)
        
        for _ in range(num_stakes):
            # Calculate stake amount (lognormal distribution)
            amount = self.rng.lognormal(
                mean=np.log(self.state.total_supply * 0.01),
                sigma=0.5
            )
            
            # Calculate lock duration based on expected returns
            r = returns * market_factor
            duration_probs = np.array([
                np.exp(r * (d/self.params.max_lock_duration)**self.params.gamma)
                for d in range(self.params.min_lock_duration, self.params.max_lock_duration + 1)
            ])
            duration_probs /= duration_probs.sum()
            
            duration = self.rng.choice(
                range(self.params.min_lock_duration, self.params.max_lock_duration + 1),
                p=duration_probs
            )
            
            try:
                self.create_lock(amount, duration)
            except ValueError:
                continue  
    
    def _process_fees(self) -> None:
        # First generate fees as before
        market_rate = self.state.market_state.base_fee_rate
        
        # Initialize fee distribution metrics for this epoch
        fee_metrics = {
            'total_distributable_fees': 0.0,
            'total_distributed_fees': 0.0,
            'fees_per_pod': {},
            'active_claims_per_pod': {},
            'distributed_fees_per_pod': {},
            'average_fee_per_fcu': {}
        }
        
        for pod_name, pod in self.state.pods.items():
            base_drift = pod.fee_drift * pod.votes * market_rate
            vol = self.params.fee_volatility * np.sqrt(pod.votes) * market_rate
            fee_change = self.rng.normal(base_drift, vol)
            
            old_fees = pod.fees
            pod.fees += fee_change
            if pod.fees < 0:
                pod.fees = 0

            # Track pod-specific fees
            fee_metrics['fees_per_pod'][pod_name] = pod.fees
            
            # Now distribute fees to active FCU holders
            if not hasattr(pod, 'active_claims'):
                pod.active_claims = []
            
            total_active_fcus = sum(claim.amount for claim in pod.active_claims)
            fee_metrics['active_claims_per_pod'][pod_name] = total_active_fcus
            
            if total_active_fcus > 0:
                distributable_fees = pod.fees * pod.fee_split
                fee_metrics['total_distributable_fees'] += distributable_fees
                distributed_fees = 0.0
                
                for claim in pod.active_claims:
                    position = self.state.lock_positions[claim.staker_index]
                    if not position.is_expired(self.state.epoch):
                        fee_share = distributable_fees * (claim.amount / total_active_fcus)
                        if not hasattr(position, 'earned_fees'):
                            position.earned_fees = defaultdict(float)
                        position.earned_fees[pod_name] += fee_share
                        distributed_fees += fee_share
                
                fee_metrics['distributed_fees_per_pod'][pod_name] = distributed_fees
                fee_metrics['average_fee_per_fcu'][pod_name] = (
                    distributed_fees / total_active_fcus if total_active_fcus > 0 else 0
                )
                fee_metrics['total_distributed_fees'] += distributed_fees

            pod.update_metrics()

        # Update state metrics with fee distribution data
        self.state.metrics.update({
            'fee_distribution': fee_metrics,
            'total_fees_distributed': fee_metrics['total_distributed_fees'],
            'total_distributable_fees': fee_metrics['total_distributable_fees'],
            'fee_distribution_ratio': (
                fee_metrics['total_distributed_fees'] / fee_metrics['total_distributable_fees'] 
                if fee_metrics['total_distributable_fees'] > 0 else 0
            ),
            'pod_fee_metrics': {
                pod_name: {
                    'total_fees': fee_metrics['fees_per_pod'].get(pod_name, 0),
                    'active_fcus': fee_metrics['active_claims_per_pod'].get(pod_name, 0),
                    'distributed_fees': fee_metrics['distributed_fees_per_pod'].get(pod_name, 0),
                    'avg_fee_per_fcu': fee_metrics['average_fee_per_fcu'].get(pod_name, 0)
                }
                for pod_name in self.state.pods.keys()
            }
        })

    def _calculate_emission(self) -> float:
        if not self.params.emission_schedule:
            raise ValueError("Emission schedule not provided")
                
        base_emission = self.params.emission_schedule(self.state)
        market_adjusted_emission = base_emission * self.state.market_state.get_market_factor()
        
        # Update emissions metrics
        self.state.metrics['emissions']['current'] = market_adjusted_emission
        self.state.metrics['emissions']['total'] += market_adjusted_emission
        
        # Handle vesting if enabled
        if self.vesting_manager:
            self.vesting_manager.add_emission(market_adjusted_emission, self.state.epoch)
            vested_amount = self.vesting_manager.tokens_to_release(self.state.epoch)
            self.state.metrics['emissions']['vested'] += vested_amount
            self.state.metrics['emissions']['unvested'] = (
                self.vesting_manager.remaining_tokens(self.state.epoch)
            )
            return vested_amount
        
        return market_adjusted_emission
    
    # TODO: votes can only be updated with NEW Reppo tokens, which are staked and converted to veReppo
    #       stakes cannot be updated.  
    #       Decide how we want to do this - right now, only pods basically accrue new reppo tokens, which can 
    #       then be used to restake into veReppo.  However, we want to have a notion of "users" voting.  is that
    #       separate from pods voting, and do we want the simulation to handle both?
    def _update_votes(self) -> None:
        print(f"\nEpoch {self.state.epoch} Vote Update:")
        total_fees = sum(pod.fees for pod in self.state.pods.values())
        print(f"Total Fees: {total_fees:.4f}")
        
        if total_fees == 0:
            vote_weights = np.full(len(self.state.pods), 1.0 / len(self.state.pods))
            print("No fees - using equal vote distribution")
        else:
            vote_weights = np.array([
                self.params.alpha * (pod.fees / total_fees) + self.params.delta
                for pod in self.state.pods.values()
            ])
            vote_weights = vote_weights / np.sum(vote_weights)
            print("Performance-based vote distribution")
        
        for pod_name, (pod, weight) in zip(self.state.pods.keys(), zip(self.state.pods.values(), vote_weights)):
            old_votes = pod.votes
            pod.votes = weight
            print(f"{pod_name}: {old_votes:.4f} -> {weight:.4f}")
    
    # TODO: remove expiration of FCUs.
    def _generate_fcus(self) -> None:
        for pod in self.state.pods.values():
            new_fcus = pod.fcu_generation_rate * pod.votes
            pod.fcus += float(new_fcus)

        # Generate individual FCU claims for stakers
        for i, position in enumerate(self.state.lock_positions):
            if position.is_expired(self.state.epoch):
                continue
                
            ve_power = position.ve_power(self.params)
            
            # Calculate FCUs earned for each lock position based on vote allocation
            for pod_name, pod in self.state.pods.items():
                vote_share = pod.votes * ve_power / self.state.ve_tokens
                fcu_amount = pod.fcu_generation_rate * vote_share
                
                if fcu_amount > 0:
                    activation_epoch = (
                        self.state.epoch + 
                        self.params.fcu_delay[pod_name] + 
                        1
                    )
                    expiry_epoch = activation_epoch + self.params.fcu_duration
                    
                    claim = FCUClaim(
                        staker_index=i,
                        pod_name=pod_name,
                        amount=fcu_amount,
                        acquisition_epoch=self.state.epoch,
                        activation_epoch=activation_epoch,
                        expiry_epoch=expiry_epoch
                    )
                    
                    position.fcu_claims[pod_name].append(claim)
                    
                    # Schedule activation
                    self.events.append(Event(
                        type=EventType.FCU_ACTIVATION,
                        epoch=activation_epoch,
                        data={"claim": claim}
                    ))

    def _update_ve_tokens(self) -> None:
        self.state.ve_tokens = sum(
            position.ve_power(self.params)
            for position in self.state.lock_positions
            if not position.is_expired(self.state.epoch)
        )

    def _record_state(self) -> None:
        metrics = self.state.metrics.copy()
        if self.vesting_manager:
            metrics['unvested_emissions'] = self.vesting_manager.remaining_tokens(self.state.epoch)
        
        self.history.append({
            'epoch': self.state.epoch,
            'total_supply': self.state.total_supply,
            'locked_tokens': self.state.locked_tokens,
            've_tokens': self.state.ve_tokens,
            'market_rate': self.state.market_state.base_fee_rate,
            'metrics': self.state.metrics.copy(),  # Now contains all metrics in organized structure
        })

    def _check_pod_performance(self) -> None:
        median_fees = np.median([pod.fees for pod in self.state.pods.values()])
        
        for pod in self.state.pods.values():
            if not hasattr(pod, 'consecutive_good_epochs'):
                pod.consecutive_good_epochs = 0
                
            if pod.fees >= median_fees:
                pod.consecutive_good_epochs += 1
            else:
                pod.consecutive_good_epochs = 0
                
            # Update emission multiplier based on performance
            if pod.consecutive_good_epochs == 0:
                pod.emission_multiplier = 0.25
            elif pod.consecutive_good_epochs == 1:
                pod.emission_multiplier = 0.50
            elif pod.consecutive_good_epochs == 2:
                pod.emission_multiplier = 0.75
            else:
                pod.emission_multiplier = 1.00  # Graduated

    def _process_emission(self) -> None:
        emission = self._calculate_emission()
        self.state.total_supply += emission
        
        # Clean up completed vesting schedules
        if self.vesting_manager:
            self.vesting_manager.cleanup_completed(self.state.epoch)

        # Track the emission amount in new metrics structure
        self.state.metrics['emissions']['current'] = emission
        
        # Distribute the vested/released emissions to pods
        total_votes = sum(pod.votes for pod in self.state.pods.values())
        if total_votes == 0:
            emissions_per_pod = emission / len(self.state.pods)
            print("No votes - distributing emissions equally")
            for pod in self.state.pods.values():
                pod.emissions = emissions_per_pod
        else:
            print("Vote-weighted emission distribution:")
            for name, pod in self.state.pods.items():
                pod.emissions = emission * (pod.votes / total_votes)
                print(f"  {name}: {pod.emissions:.4f} (votes: {pod.votes:.4f})")

    def create_lock(self, amount: float, duration: int) -> None:
        if amount <= 0 or duration <= 0:
            raise ValueError("Amount and duration must be positive")
        if duration > self.params.max_lock_duration:
            raise ValueError("Duration exceeds maximum lock duration")
        if duration < self.params.min_lock_duration:
            raise ValueError("Duration below minimum lock duration")
        if amount > self.state.total_supply - self.state.locked_tokens:
            raise ValueError("Insufficient unlocked tokens")
            
        position = LockPosition(
            amount=amount,
            duration=duration,
            start_epoch=self.state.epoch
        )
        
        self.state.lock_positions.append(position)
        self.state.locked_tokens += amount
        self._update_ve_tokens()
        
        self.events.append(Event(
            type=EventType.LOCK_EXPIRY,
            epoch=self.state.epoch + duration,
            data={"position_index": len(self.state.lock_positions) - 1}
        ))
    
    def _process_lock_expiry(self, position_index: int) -> None:
        position = self.state.lock_positions[position_index]
        self.state.locked_tokens -= position.amount
        self._update_ve_tokens()
        
    def extend_lock(self, position_index: int, new_duration: int) -> None:
        position = self.state.lock_positions[position_index]
        if position.is_expired(self.state.epoch):
            raise ValueError("Cannot extend expired lock")
        
        old_ve_power = position.ve_power(self.params)
        position.extend_duration(new_duration, self.params)
        
        self.events = [e for e in self.events 
                      if not (e.type == EventType.LOCK_EXPIRY and 
                             e.data["position_index"] == position_index)]
        
        self.events.append(Event(
            type=EventType.LOCK_EXPIRY,
            epoch=self.state.epoch + position.duration,
            data={"position_index": position_index}
        ))
        
        self._update_ve_tokens()

    def _process_fcu_activation(self, event_data: dict) -> None:
        """Activates an FCU claim when it becomes eligible for fee distribution"""
        claim = event_data["claim"]
        position = self.state.lock_positions[claim.staker_index]
        pod = self.state.pods[claim.pod_name]
        
        # Mark claim as active by moving it to active claims
        if not hasattr(pod, 'active_claims'):
            pod.active_claims = []
        pod.active_claims.append(claim)
        
        # Schedule claim expiry
        self.events.append(Event(
            type=EventType.FCU_EXPIRY,
            epoch=claim.expiry_epoch,
            data={"claim": claim}
        ))

    # includes market correlation ... can activate after initial validation of current code
    # def _process_fees(self) -> None:
    #     market_rate = self.state.market_state.base_fee_rate
    #     num_pods = len(self.state.pods)
        
    #     # Generate correlated normal random variables
    #     correlations = self.params.pod_correlation_matrix
    #     means = np.array([
    #         pod.fee_drift * pod.votes * market_rate 
    #         for pod in self.state.pods.values()
    #     ])
    #     vols = np.array([
    #         self.params.fee_volatility * np.sqrt(pod.votes) * market_rate
    #         for pod in self.state.pods.values()
    #     ])
        
    #     # Create covariance matrix
    #     cov = np.outer(vols, vols) * correlations
        
    #     # Generate correlated fee changes
    #     fee_changes = self.rng.multivariate_normal(means, cov)
        
    #     # Apply fee changes to pods
    #     for pod, fee_change in zip(self.state.pods.values(), fee_changes):
    #         pod.fees += fee_change
    #         if pod.fees < 0:
    #             pod.fees = 0
    #         pod.update_metrics()

    # def check_system_stability(self) -> Dict[str, bool]:
    #     total_fees = self.state.get_total_fees()
    #     total_emissions = self.state.metrics['emissions_this_epoch']
    #     max_vote_share = max(pod.votes for pod in self.state.pods.values())
        
    #     stability = {
    #         'fee_coverage': total_fees >= self.params.min_fee_coverage * total_emissions,
    #         'vote_distribution': max_vote_share <= self.params.max_vote_concentration,
    #         'pod_viability': all(
    #             pod.total_fees >= self.params.min_viable_fees 
    #             for pod in self.state.pods.values()
    #         ),
    #         'tvl_growth': self.state.locked_tokens > self.state._prev_locked_tokens
    #             if hasattr(self.state, '_prev_locked_tokens') else True
    #     }
        
    #     self.state._prev_locked_tokens = self.state.locked_tokens
    #     return stability

@dataclass 
class StakingConfig:
    amount_per_epoch: float
    duration: int
    
@dataclass
class PodConfig:
    fee_growth: float  # Fixed fee growth per epoch
    initial_vote_share: float

@dataclass
class DeterministicConfig:
    staking: StakingConfig
    pods: Dict[str, PodConfig]
    market_growth: float  # Fixed market growth per epoch

class DeterministicSimulation(VeTokenomicsSimulation):
    def __init__(self, params: SimulationParams, det_config: DeterministicConfig):
        super().__init__(params)
        self.det_config = det_config
        
        # Initialize pod vote shares
        for pod_name, pod in self.state.pods.items():
            pod.votes = self.det_config.pods[pod_name].initial_vote_share
        
    def _simulate_market(self) -> None:
        self.state.market_state.base_fee_rate *= (1 + self.det_config.market_growth)
    
    def _simulate_staking(self) -> None:
        try:
            self.create_lock(
                self.det_config.staking.amount_per_epoch,
                self.det_config.staking.duration
            )
        except ValueError:
            pass
    
    def _process_fees(self) -> None:
        market_rate = self.state.market_state.base_fee_rate
        
        for pod_name, pod in self.state.pods.items():
            pod_config = self.det_config.pods[pod_name]
            if pod.fees == 0:
                pod.fees = 1  # seed
            pod.fees *= (1 + pod_config.fee_growth)
            pod.update_metrics()