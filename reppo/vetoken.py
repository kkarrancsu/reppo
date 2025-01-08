from typing import Dict, List, Optional, NamedTuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from numpy.random import Generator, PCG64

class EventType(Enum):
    LOCK_EXPIRY = "lock_expiry"
    FCU_ACTIVATION = "fcu_activation" 
    EMISSION_DIST = "emission_dist"
    PERF_CHECK = "perf_check"
    FEE_DIST = "fee_dist"

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
    
    def apply_market_shock(self, shock: float) -> None:
        self.base_fee_rate *= (1 + shock)
        if self.base_fee_rate < 0:
            self.base_fee_rate = 0
            
    def apply_growth(self) -> None:
        self.base_fee_rate *= (1 + self.growth_rate)

    def get_market_factor(self, initial_rate: float) -> float:
        return self.base_fee_rate / initial_rate if initial_rate > 0 else 1.0

class VestingSchedule:
    def __init__(
        self,
        initial_amount: float,
        cliff_epoch: int,
        vesting_duration: int,
        release_frequency: int
    ) -> None:
        if initial_amount <= 0:
            raise ValueError("Initial amount must be positive")
        if cliff_epoch < 0:
            raise ValueError("Cliff epoch cannot be negative")
        if vesting_duration <= 0:
            raise ValueError("Vesting duration must be positive")
        if release_frequency <= 0:
            raise ValueError("Release frequency must be positive")
        if release_frequency > vesting_duration:
            raise ValueError("Release frequency cannot exceed vesting duration")

        self.initial_amount = initial_amount
        self.cliff_epoch = cliff_epoch
        self.vesting_duration = vesting_duration
        self.release_frequency = release_frequency
        
        self.release_epochs = []
        self.amounts = []
        self._initialize_schedule()
    
    def _initialize_schedule(self) -> None:
        num_releases = self.vesting_duration // self.release_frequency
        amount_per_release = self.initial_amount / num_releases
        
        for i in range(num_releases):
            epoch = self.cliff_epoch + (i * self.release_frequency)
            self.release_epochs.append(epoch)
            self.amounts.append(amount_per_release)
    
    def tokens_to_release(self, epoch: int) -> float:
        if epoch < self.cliff_epoch:
            return 0.0
        
        if epoch in self.release_epochs:
            index = self.release_epochs.index(epoch)
            return self.amounts[index]
        
        return 0.0

    def remaining_tokens(self, epoch: int) -> float:
        if epoch < self.cliff_epoch:
            return self.initial_amount
            
        released = sum(amount for e, amount in zip(self.release_epochs, self.amounts) 
                      if e <= epoch)
        return self.initial_amount - released
    
class SystemState:
    def __init__(self, params):
        self.epoch = 0
        self.total_supply = params.initial_token_supply
        self.locked_tokens = 0.0
        self.ve_tokens = 0.0
        self.lock_positions: List[LockPosition] = []

        # Initialize pods with equal vote share
        initial_vote_share = 1.0 / len(params.initial_pods)
        print(f"\nInitializing pods with vote share: {initial_vote_share}")
        

        self.pods: Dict[str, PodState] = {
            pod: PodState(0.0, 1.0 / len(params.initial_pods), 0.0, params.base_fee_drift)
            for pod in params.initial_pods
        }
        print("\nInitial Pod States:")
        for name, pod in self.pods.items():
            print(f"{name}:")
            print(f"  Votes: {pod.votes:.4f}")
            print(f"  Fee Drift: {pod.fee_drift:.4f}")

        self.market_state = params.market
        self.metrics = {
            'avg_lock_duration': 0.0,
            'total_fees': 0.0,
            'total_fcus': 0.0,
            'vote_entropy': 0.0,
            'active_positions': 0,
            'emissions_this_epoch': 0.0,
            'total_emissions': 0.0,
            'total_vested': 0.0
        }
    
    def update_metrics(self) -> None:
        active_positions = [p for p in self.lock_positions if not p.is_expired(self.epoch)]
        if active_positions:
            self.metrics['avg_lock_duration'] = np.mean([p.duration for p in active_positions])
            self.metrics['active_positions'] = len(active_positions)
        
        self.metrics['total_fees'] = sum(pod.total_fees for pod in self.pods.values())
        self.metrics['total_fcus'] = sum(pod.total_fcus for pod in self.pods.values())
        
        votes = np.array([pod.votes for pod in self.pods.values()])
        votes = votes[votes > 0]
        if len(votes) > 0:
            self.metrics['vote_entropy'] = -np.sum(votes * np.log(votes))
            
        # Track pod-specific metrics
        self.metrics['pod_emissions'] = {name: pod.emissions for name, pod in self.pods.items()}
        self.metrics['total_pod_emissions'] = {name: pod.total_emissions for name, pod in self.pods.items()}
        
        # Current epoch metrics
        self.metrics['pod_fees'] = {name: pod.fees for name, pod in self.pods.items()}
        self.metrics['pod_fcus'] = {name: pod.fcus for name, pod in self.pods.items()}
        self.metrics['vote_distribution'] = {name: pod.votes for name, pod in self.pods.items()}
        
        # Historical aggregates
        self.metrics['cumulative_pod_fees'] = {name: pod.total_fees for name, pod in self.pods.items()}
        self.metrics['cumulative_pod_fcus'] = {name: pod.total_fcus for name, pod in self.pods.items()}
        self.metrics['avg_vote_share'] = {name: pod.avg_vote_share for name, pod in self.pods.items()}
        
        # Performance metrics
        self.metrics['fee_generation_rate'] = {
            name: pod.fees / max(pod.votes, 0.0001) 
            for name, pod in self.pods.items()
        }
        self.metrics['fcu_efficiency'] = {
            name: pod.fcus / max(pod.fees, 0.0001) 
            for name, pod in self.pods.items()
        }
        
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
    vesting: Optional[VestingSchedule] = None
    emission_schedule: Optional[Callable[[SystemState], float]] = None
    
    # # features to be included soon...
    # min_viable_fees: float 
    # max_vote_concentration: float
    # min_fee_coverage: float
    # pod_correlation_matrix: np.ndarray = np.eye(len(initial_pods))
@dataclass
class LockPosition:
    amount: float
    duration: int
    start_epoch: int
    
    def is_expired(self, current_epoch: int) -> bool:
        return current_epoch >= self.start_epoch + self.duration
        
    def ve_power(self, params: SimulationParams) -> float:
        duration_weight = (self.duration / params.max_lock_duration) ** params.gamma
        return self.amount * duration_weight
        
    def extend_duration(self, new_duration: int) -> None:
        if new_duration <= self.duration:
            raise ValueError("New duration must be greater than current duration")
        self.duration = new_duration

# TODO: emissions vesting!
class VeTokenomicsSimulation:
    def __init__(self, params: SimulationParams):
        self.params = params
        self.state = SystemState(params)
        self.events: List[Event] = []
        self.rng = np.random.Generator(PCG64())
        self.history: List[dict] = []
        self._init_events()
    
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
    
    def _simulate_market(self) -> None:
        growth = self.state.market_state.growth_rate
        vol = self.state.market_state.volatility
        shock = self.rng.normal(0, vol)
        
        self.state.market_state.base_fee_rate *= np.exp(growth + shock)
    
    def _calculate_emission(self) -> float:
        if self.params.emission_schedule:
            base_emission = self.params.emission_schedule(self.state)
        else:
            raise ValueError("Emission schedule not provided")
            
        market_factor = self.state.market_state.base_fee_rate / self.params.market.base_fee_rate
        emission = base_emission * market_factor
        
        # Add vested tokens if there's a vesting schedule
        vested_amount = 0.0
        if self.params.vesting:
            vested_amount = self.params.vesting.tokens_to_release(self.state.epoch)
            self.state.metrics['total_vested'] += vested_amount
            
        total_emission = emission + vested_amount
        self.state.metrics['emissions_this_epoch'] = total_emission
        self.state.metrics['total_emissions'] += emission  # Track only base emissions
        
        return total_emission
    
    def _process_emission(self) -> None:
        emission = self._calculate_emission()
        self.state.total_supply += emission
        
        print(f"\nEpoch {self.state.epoch} Emission Distribution:")
        print(f"Total Emission: {emission:.4f}")
        
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
    
    def _process_fees(self) -> None:
        print(f"\nEpoch {self.state.epoch} Fee Processing:")
        market_rate = self.state.market_state.base_fee_rate
        print(f"Market Rate: {market_rate:.4f}")
        
        for pod_name, pod in self.state.pods.items():
            base_drift = pod.fee_drift * pod.votes * market_rate
            vol = self.params.fee_volatility * np.sqrt(pod.votes) * market_rate
            fee_change = self.rng.normal(base_drift, vol)
            
            old_fees = pod.fees
            pod.fees += fee_change
            if pod.fees < 0:
                pod.fees = 0
            
            print(f"{pod_name}:")
            print(f"  Votes: {pod.votes:.4f}")
            print(f"  Base Drift: {base_drift:.4f}")
            print(f"  Volatility: {vol:.4f}")
            print(f"  Fee Change: {fee_change:.4f}")
            print(f"  Fees: {old_fees:.4f} -> {pod.fees:.4f}")
            
            pod.update_metrics()

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
    
    def _record_state(self) -> None:
        self.history.append({
            'epoch': self.state.epoch,
            'total_supply': self.state.total_supply,
            'locked_tokens': self.state.locked_tokens,
            've_tokens': self.state.ve_tokens,
            'metrics': self.state.metrics.copy(),
            'market_rate': self.state.market_state.base_fee_rate,
            'emissions_this_epoch': self.state.metrics['emissions_this_epoch'],
            'total_emissions': self.state.metrics['total_emissions'],
            'total_vested': self.state.metrics['total_vested'],
            'pod_emissions': self.state.metrics['pod_emissions'],
            'total_pod_emissions': self.state.metrics['total_pod_emissions'],
            'pod_fees': self.state.metrics['pod_fees'],
            'pod_fcus': self.state.metrics['pod_fcus'],
            'vote_distribution': self.state.metrics['vote_distribution'],
            'total_fees': self.state.metrics['total_fees'],
            'total_fcus': self.state.metrics['total_fcus'],
            'avg_lock_duration': self.state.metrics['avg_lock_duration'],
            'active_positions': self.state.metrics['active_positions'],
            'vote_entropy': self.state.metrics['vote_entropy']
        })

    def _simulate_staking(self) -> None:
        base_rate = self.params.base_stake_rate  # Base staking rate
        
        # Calculate factors that influence staking rate
        returns = self.state.get_total_fees() / max(self.state.locked_tokens, 1)
        market_factor = self.state.market_state.get_market_factor(self.params.market.base_fee_rate)
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
        
        # stability = self.check_system_stability()
        # self.state.metrics['stability'] = stability
        
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
    
    def _init_events(self) -> None:
        self.events.append(Event(
            type=EventType.EMISSION_DIST,
            epoch=1,
            data={}
        ))

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
                
            elif event.type == EventType.FEE_DIST:
                self._process_fees()
                
            elif event.type == EventType.LOCK_EXPIRY:
                self._process_lock_expiry(event.data["position_index"])
        self.events = [e for e in self.events if e.epoch != self.state.epoch]
        
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
            
    def _generate_fcus(self) -> None:
        for pod in self.state.pods.values():
            pod.fcus += self.params.omega * pod.fees * pod.votes
            
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
        
    def _update_ve_tokens(self) -> None:
        self.state.ve_tokens = sum(
            position.ve_power(self.params)
            for position in self.state.lock_positions
            if not position.is_expired(self.state.epoch)
        )
        
    def run(self, epochs: Optional[int] = None) -> List[SystemState]:
        max_epochs = epochs or self.params.epochs
        states = [self.state]
        
        for _ in range(max_epochs):
            state = self.step()
            states.append(state)
            
        return states
    
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