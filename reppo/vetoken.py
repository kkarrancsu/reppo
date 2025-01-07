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
        self.pods: Dict[str, PodState] = {
            pod: PodState(0.0, 1.0 / len(params.initial_pods), 0.0, params.base_fee_drift)
            for pod in params.initial_pods
        }
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
    
    # Historical metrics
    total_fees: float = 0.0
    total_fcus: float = 0.0
    peak_votes: float = 0.0
    min_votes: float = 1.0
    cumulative_vote_share: float = 0.0
    vote_samples: int = 0
    
    def update_metrics(self) -> None:
        self.total_fees += self.fees
        self.total_fcus += self.fcus
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
    base_stake_rate: float
    fee_volatility: float
    correlation_matrix: np.ndarray
    base_fee_drift: float
    max_lock_duration: int
    min_lock_duration: int  # Added minimum lock duration
    initial_pods: List[str]
    initial_token_supply: float
    epochs: int
    market: MarketState
    vesting: Optional[VestingSchedule] = None
    emission_schedule: Optional[Callable[[SystemState], float]] = None

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
    
    def _process_fees(self) -> None:
        market_rate = self.state.market_state.base_fee_rate
        
        for pod in self.state.pods.values():
            base_drift = pod.fee_drift * pod.votes * market_rate
            vol = self.params.fee_volatility * np.sqrt(pod.votes) * market_rate
            
            pod.fees += self.rng.normal(base_drift, vol)
            if pod.fees < 0:
                pod.fees = 0
                
            pod.update_metrics()
    
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
            'total_vested': self.state.metrics['total_vested']
        })
    
    def step(self) -> SystemState:
        self._process_events()
        self._simulate_market()
        self._process_fees()
        self._update_votes()
        self._generate_fcus()
        self._update_ve_tokens()
        
        self.state.update_metrics()
        self._record_state()
        
        self.state.epoch += 1
        return self.state
    
    def _init_events(self) -> None:
        self.events.append(Event(
            type=EventType.EMISSION_DIST,
            epoch=1,
            data={}
        ))
        
    def _process_events(self) -> None:
        current_events = [e for e in self.events if e.epoch == self.state.epoch]
        for event in current_events:
            self._handle_event(event)
        self.events = [e for e in self.events if e.epoch != self.state.epoch]
            
    def _handle_event(self, event: Event) -> None:
        if event.type == EventType.EMISSION_DIST:
            self._process_emission()
            # Schedule next emission
            self.events.append(Event(
                type=EventType.EMISSION_DIST,
                epoch=self.state.epoch + 1,
                data={}
            ))
        elif event.type == EventType.FEE_DIST:
            self._process_fees()
        elif event.type == EventType.LOCK_EXPIRY:
            self._process_lock_expiry(event.data["position_index"])
            
    def _process_emission(self) -> None:
        emission = self._calculate_emission()
        self.state.total_supply += emission
        
    def _process_fees(self) -> None:
        for pod in self.state.pods.values():
            drift = pod.fee_drift * pod.votes
            volatility = self.params.fee_volatility * np.sqrt(pod.votes)
            pod.fees += self.rng.normal(drift, volatility)
            
    def _update_votes(self) -> None:
        total_fees = self.state.get_total_fees()
        if total_fees == 0:
            return
            
        vote_weights = np.array([
            self.params.alpha * (pod.fees / total_fees) + self.params.delta
            for pod in self.state.pods.values()
        ])
        vote_weights = vote_weights / np.sum(vote_weights)
        
        for pod, weight in zip(self.state.pods.values(), vote_weights):
            pod.votes = weight
            
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

    def _calculate_emission(self) -> float:
        if self.params.emission_schedule:
            base_emission = self.params.emission_schedule(self.state)
        else:
            # Default emission schedule if none provided
            base_emission = 100 * (0.95 ** self.state.epoch)
            
        market_factor = self.state.market_state.base_fee_rate / self.params.market.base_fee_rate
        
        # Add vested tokens if there's a vesting schedule
        vested_amount = 0.0
        if self.params.vesting:
            vested_amount = self.params.vesting.tokens_to_release(self.state.epoch)
            
        return (base_emission * market_factor) + vested_amount
        
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