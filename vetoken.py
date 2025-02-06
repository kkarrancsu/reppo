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
class TokenBuyConfig:
    base_buy_rate: float
    market_sensitivity: float
    randomization_factor: float

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
        
        self.pods: Dict[str, PodState] = {
            pod: PodState(0.0, 1.0 / len(params.initial_pods), 0.0, params.base_fee_drift)
            for pod in params.initial_pods
        }
        
        self.market_state = params.market
        self.metrics = {
            'avg_lock_duration': 0.0,
            'active_users': 0,
            'vote_entropy': 0.0,
            'emissions': {
                'current': 0.0,
                'total': 0.0,
                'vested': 0.0,
                'unvested': 0.0
            },
            'pods': {}
        }

    def update_metrics(self, users: List) -> None:
        active_users = [u for u in users if u.is_locked(self.epoch)]
        if active_users:
            self.metrics['avg_lock_duration'] = float(np.mean([u.lock_duration for u in active_users]))
            self.metrics['active_users'] = len(active_users)
        
        pod_metrics = {}
        for name, pod in self.pods.items():
            active_claims = getattr(pod, 'active_claims', [])
            
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
                    'active': len(active_claims)
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
    lock_duration: int      # Fixed lock duration for all positions
    lock_interval: int      # How often new locks can be created
    max_lock_duration: int
    initial_pods: List[str]
    initial_token_supply: float
    epochs: int
    market: MarketState

    fcu_duration: int  # τ in the paper - how long FCUs last
    fcu_delay: Dict[str, int]  # δp for each pod - delay before FCUs activate
    pod_fcu_rates: Dict[str, float]  # FCU generation rates per pod

    emission_schedule: Optional[Callable[[SystemState], float]] = None
    emission_vesting_duration: Optional[int] = None  # None means instant vesting
    
@dataclass
class FCUClaim:
    user_id: int      # User who owns the claim
    pod_name: str     # Pod the claim is for
    amount: float     # Amount of FCUs (based on their effective votes * ω)
    acquisition_epoch: int  # When earned
    activation_epoch: int   # When claim becomes active (acquisition + δp)
    expiry_epoch: int      # When claim expires (activation + τ)

@dataclass
class User:
    id: int
    token_balance: float = 0
    staked_amount: float = 0
    lock_start_epoch: int = -1
    lock_duration: int = 0
    pending_tokens: float = 0
    cached_ve_power: float = 0
    pod_votes: Dict[str, float] = field(default_factory=dict)
    fcu_claims: Dict[str, List[FCUClaim]] = field(default_factory=lambda: defaultdict(list))
    earned_fees: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    
    def is_locked(self, current_epoch: int) -> bool:
        return (self.lock_start_epoch != -1 and 
                current_epoch < self.lock_start_epoch + self.lock_duration)
    
    def can_stake(self, amount: float) -> bool:
        return amount <= self.token_balance
    
    def ve_power(self, params: SimulationParams) -> float:
        if not self.is_locked(params.current_epoch):
            return 0
        duration_weight = (self.lock_duration / params.max_lock_duration) ** params.gamma
        return self.staked_amount * duration_weight
    
class VeTokenomicsSimulation:
    def __init__(self, params: SimulationParams):
        self.params = params
        self.state = SystemState(params)
        self.events: List[Event] = []
        self.rng = np.random.Generator(PCG64())
        self.history: List[dict] = []
        self.users: List[User] = []
        if params.emission_vesting_duration:
            self.vesting_manager = VestingManager(params.emission_vesting_duration)
        else:
            self.vesting_manager = None
        self._init_events()

    def create_user(self) -> int:
        user = User(id=len(self.users))
        self.users.append(user)
        return user.id
    
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
    
    def _simulate_market(self) -> None:
        growth = self.state.market_state.growth_rate
        vol = self.state.market_state.volatility
        shock = self.rng.normal(0, vol)
        
        self.state.market_state.base_fee_rate *= np.exp(growth + shock)
    
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
        
        self.state.update_metrics(self.users)
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
                self._process_emissions()
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

            elif event.type == EventType.FCU_EXPIRY:
                self._process_fcu_expiry(event.data)

        self.events = [e for e in self.events if e.epoch != self.state.epoch]
    
    def stake_tokens(self, user_id: int, amount: float) -> None:
        user = self.users[user_id]
        if not user.can_stake(amount):
            raise ValueError("Insufficient balance")
        if user.is_locked(self.state.epoch):
            raise ValueError("Already has active lock")
            
        user.token_balance -= amount
        user.staked_amount = amount
        user.lock_start_epoch = self.state.epoch
        user.lock_duration = self.params.lock_duration
        user.cached_ve_power = user.ve_power(self.params)
        
        # Randomly allocate votes across pods initially
        total_power = user.cached_ve_power
        vote_weights = self.rng.dirichlet([1.0] * len(self.state.pods))
        
        for pod_name, weight in zip(self.state.pods.keys(), vote_weights):
            user.pod_votes[pod_name] = total_power * weight
        
        self.state.locked_tokens += amount
        self._update_ve_tokens()
        
    def _simulate_staking(self) -> None:
        if self.state.epoch % self.params.lock_interval != 0:
            return
        
        self._process_user_token_buys()
        
        # Then allow users to stake
        for user in self.users:
            if user.is_locked(self.state.epoch):
                continue
                
            stake_amount = min(
                user.token_balance,
                user.token_balance * self.rng.uniform(0.5, 1.0)
            )
            
            if stake_amount > 0:
                try:
                    self.stake_tokens(user.id, stake_amount)
                except ValueError:
                    continue
        
    def _process_fees(self) -> None:
        market_rate = self.state.market_state.base_fee_rate
        fee_metrics = {
            'total_distributable_fees': 0.0,
            'total_distributed_fees': 0.0,
            'fees_per_pod': {},
            'active_claims_per_pod': {},
            'distributed_fees_per_pod': {},
            'average_fee_per_fcu': {}
        }
        
        for pod_name, pod in self.state.pods.items():
            # Generate fees
            self._generate_pod_fees(pod, market_rate)
            fee_metrics['fees_per_pod'][pod_name] = pod.fees

            # Distribute fees to FCU holders
            self._distribute_fees_to_fcus(pod, pod_name, fee_metrics)
            pod.update_metrics()

        self.state.metrics.update({
            'fee_distribution': fee_metrics,
            'total_fees_distributed': fee_metrics['total_distributed_fees'],
            'total_distributable_fees': fee_metrics['total_distributable_fees'],
            'fee_distribution_ratio': (
                fee_metrics['total_distributed_fees'] / fee_metrics['total_distributable_fees'] 
                if fee_metrics['total_distributable_fees'] > 0 else 0
            )
        })

    def _generate_pod_fees(self, pod, market_rate: float) -> None:
        base_drift = pod.fee_drift * pod.votes * market_rate
        vol = self.params.fee_volatility * np.sqrt(pod.votes) * market_rate
        fee_change = self.rng.normal(base_drift, vol)
        
        pod.fees += fee_change
        if pod.fees < 0:
            pod.fees = 0

    def _distribute_fees_to_fcus(self, pod, pod_name: str, fee_metrics: dict) -> None:
        if not hasattr(pod, 'active_claims'):
            pod.active_claims = []

        total_active_fcus = sum(claim.amount for claim in pod.active_claims)
        fee_metrics['active_claims_per_pod'][pod_name] = total_active_fcus

        if total_active_fcus > 0:
            distributable_fees = pod.fees * pod.fee_split
            fee_metrics['total_distributable_fees'] += distributable_fees
            distributed_fees = self._distribute_fees_to_claims(
                pod.active_claims, distributable_fees, total_active_fcus, pod_name
            )
            
            fee_metrics['distributed_fees_per_pod'][pod_name] = distributed_fees
            fee_metrics['average_fee_per_fcu'][pod_name] = (
                distributed_fees / total_active_fcus if total_active_fcus > 0 else 0
            )
            fee_metrics['total_distributed_fees'] += distributed_fees

    def _distribute_fees_to_claims(
        self, 
        claims: List[FCUClaim], 
        distributable_fees: float,
        total_active_fcus: float,
        pod_name: str
    ) -> float:
        distributed_fees = 0.0
        for claim in claims:
            user = self.users[claim.user_id]
            if user.is_locked(self.state.epoch):
                fee_share = distributable_fees * (claim.amount / total_active_fcus)
                user.earned_fees[pod_name] += fee_share
                distributed_fees += fee_share
        return distributed_fees

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
    
    def _update_votes(self) -> None:
        if self.state.epoch % self.params.lock_duration != 0:
            return
        
        print(f"\nEpoch {self.state.epoch} Vote Update:")
        total_fees = sum(pod.fees for pod in self.state.pods.values())
        print(f"Total Fees: {total_fees:.4f}")

        # Now distribute votes proportional to fees but scaled by total vePower
        if total_fees == 0:
            vote_weights = np.full(len(self.state.pods), 1.0 / len(self.state.pods))
        else:
            vote_weights = np.array([
                self.params.alpha * (pod.fees / total_fees) + self.params.delta
                for pod in self.state.pods.values()
            ])
        vote_weights = vote_weights / np.sum(vote_weights) * self.state.ve_tokens

        for pod_name, (pod, weight) in zip(self.state.pods.keys(), zip(self.state.pods.values(), vote_weights)):
            old_votes = pod.votes
            pod.votes = weight
            print(f"{pod_name}: {old_votes:.4f} -> {weight:.4f}")
    
    def _generate_fcus(self) -> None:
        for pod in self.state.pods.values():
            new_fcus = pod.fcu_generation_rate * pod.votes
            pod.fcus += float(new_fcus)

        for user in self.users:
            if not user.is_locked(self.state.epoch):
                continue
                
            ve_power = user.cached_ve_power
            
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
                        user_id=user.id,
                        pod_name=pod_name,
                        amount=fcu_amount,
                        acquisition_epoch=self.state.epoch,
                        activation_epoch=activation_epoch,
                        expiry_epoch=expiry_epoch
                    )
                    
                    user.fcu_claims[pod_name].append(claim)
                    
                    self.events.append(Event(
                        type=EventType.FCU_ACTIVATION,
                        epoch=activation_epoch,
                        data={"claim": claim}
                    ))

    def _update_ve_tokens(self) -> None:
        self.state.ve_tokens = sum(
            user.cached_ve_power for user in self.users 
            if user.is_locked(self.state.epoch)
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

    def _process_emissions(self) -> None:
        emission = self._calculate_emission()
        self.state.total_supply += emission

        if self.vesting_manager:
            self.vesting_manager.cleanup_completed(self.state.epoch)

        self.state.metrics['emissions'].update({
            'current': emission,
            'total': self.state.metrics['emissions']['total'] + emission,
            'vested': self.vesting_manager.tokens_to_release(self.state.epoch) if self.vesting_manager else emission,
            'unvested': self.vesting_manager.remaining_tokens(self.state.epoch) if self.vesting_manager else 0
        })

    def _process_user_token_buys(self) -> None:
        if not hasattr(self, 'token_buy_config'):
            return
            
        market_factor = self.state.market_state.get_market_factor()
        base_amount = self.token_buy_config.base_buy_rate
        
        # Scale buys based on market performance
        market_adjusted = base_amount * (
            1 + self.token_buy_config.market_sensitivity * (market_factor - 1)
        )
        
        # Add randomization
        noise = self.rng.uniform(
            1 - self.token_buy_config.randomization_factor,
            1 + self.token_buy_config.randomization_factor
        )
        buy_amount = market_adjusted * noise
        
        # Distribute tokens to users
        active_users = [u for u in self.users if not u.is_locked(self.state.epoch)]
        if active_users:
            amount_per_user = buy_amount / len(active_users)
            for user in active_users:
                user.token_balance += amount_per_user

    def create_lock(self, amount: float) -> int:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if amount > self.state.total_supply - self.state.locked_tokens:
            raise ValueError("Insufficient unlocked tokens")
            
        user_id = self.create_user()
        user = self.users[user_id]
        user.token_balance = amount
        self.stake_tokens(user_id, amount)
        
        # Add lock expiry event
        self.events.append(Event(
            type=EventType.LOCK_EXPIRY,
            epoch=self.state.epoch + self.params.lock_duration,
            data={"user_id": user_id}
        ))
        
        return user_id
    
    def _process_lock_expiry(self, user_id: int) -> None:
        user = self.users[user_id]
        self.state.locked_tokens -= user.staked_amount
        user.staked_amount = 0
        user.lock_start_epoch = -1
        user.lock_duration = 0
        user.cached_ve_power = 0
        user.pod_votes.clear()
        self._update_ve_tokens()

    # TODO: determine whether this is needed, we don't need it for base functionality is my understanding  
    # def extend_lock(self, user_id: int) -> None:
    #     user = self.users[user_id]
    #     if not user.is_locked(self.state.epoch):
    #         raise ValueError("No active lock to extend")
            
    #     user.lock_duration += self.params.lock_duration
    #     user.cached_ve_power = user.ve_power(self.params)
        
    #     # Remove old expiry event and add new one
    #     self.events = [e for e in self.events 
    #                 if not (e.type == EventType.LOCK_EXPIRY and 
    #                         e.data["user_id"] == user_id)]
        
    #     self.events.append(Event(
    #         type=EventType.LOCK_EXPIRY,
    #         epoch=self.state.epoch + user.lock_duration,
    #         data={"user_id": user_id}
    #     ))
        
    #     self._update_ve_tokens()

    def _process_fcu_activation(self, event_data: dict) -> None:
        claim = event_data["claim"]
        user = self.users[claim.user_id]
        pod = self.state.pods[claim.pod_name]
        
        if not hasattr(pod, 'active_claims'):
            pod.active_claims = []
        pod.active_claims.append(claim)
        
        self.events.append(Event(
            type=EventType.FCU_EXPIRY,
            epoch=claim.expiry_epoch,
            data={"claim": claim}
        ))

    def _process_fcu_expiry(self, event_data: dict) -> None:
        claim = event_data["claim"]
        pod = self.state.pods[claim.pod_name]
        
        if hasattr(pod, 'active_claims'):
            pod.active_claims = [
                c for c in pod.active_claims 
                if not (c.user_id == claim.user_id and 
                    c.acquisition_epoch == claim.acquisition_epoch)
            ]
    
@dataclass
class PodConfig:
    fee_growth: float  # Fixed fee growth per epoch
    initial_vote_share: float

@dataclass
class DeterministicConfig:
    staking_amount_per_interval: float
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
        if self.state.epoch % self.params.lock_interval != 0:
            return
            
        try:
            self.create_lock(self.det_config.staking_amount_per_interval)
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