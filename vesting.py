from abc import ABC, abstractmethod
from typing import List

class VestingSchedule(ABC):
    def __init__(self, initial_amount: float, start_epoch: int) -> None:
        if initial_amount <= 0:
            raise ValueError("Initial amount must be positive")
        if start_epoch < 0:
            raise ValueError("Start epoch cannot be negative")
            
        self.initial_amount = initial_amount
        self.start_epoch = start_epoch

    @abstractmethod
    def tokens_to_release(self, epoch: int) -> float:
        pass

    @abstractmethod
    def remaining_tokens(self, epoch: int) -> float:
        pass

class VestingManager:
    def __init__(self, vesting_duration: int) -> None:
        self.vesting_duration = vesting_duration
        self.schedules: List[LinearVestingSchedule] = []

    def add_emission(self, amount: float, start_epoch: int) -> None:
        schedule = LinearVestingSchedule(amount, start_epoch, self.vesting_duration)
        self.schedules.append(schedule)
    
    def tokens_to_release(self, epoch: int) -> float:
        return sum(
            schedule.tokens_to_release(epoch) 
            for schedule in self.schedules
        )

    def remaining_tokens(self, epoch: int) -> float:
        return sum(
            schedule.remaining_tokens(epoch)
            for schedule in self.schedules
        )
        
    def cleanup_completed(self, current_epoch: int) -> None:
        self.schedules = [
            s for s in self.schedules 
            if current_epoch < s.start_epoch + s.duration
        ]

class LinearVestingSchedule:
    def __init__(self, initial_amount: float, start_epoch: int, vesting_duration: int) -> None:
        if initial_amount <= 0:
            raise ValueError("Initial amount must be positive")
        if vesting_duration <= 0:
            raise ValueError("Vesting duration must be positive")
            
        self.initial_amount = initial_amount
        self.start_epoch = start_epoch
        self.duration = vesting_duration
        self.amount_per_epoch = initial_amount / vesting_duration

    def tokens_to_release(self, epoch: int) -> float:
        if epoch < self.start_epoch or epoch >= self.start_epoch + self.duration:
            return 0.0
        return self.amount_per_epoch

    def remaining_tokens(self, epoch: int) -> float:
        if epoch < self.start_epoch:
            return self.initial_amount
        if epoch >= self.start_epoch + self.duration:
            return 0.0
        epochs_passed = epoch - self.start_epoch
        return self.initial_amount - (epochs_passed * self.amount_per_epoch)

class CliffVestingSchedule(VestingSchedule):
    def __init__(self, initial_amount: float, cliff_epoch: int, duration: int, release_frequency: int) -> None:
        super().__init__(initial_amount, cliff_epoch)
        if duration <= 0:
            raise ValueError("Duration must be positive")
        if release_frequency <= 0:
            raise ValueError("Release frequency must be positive")
        if release_frequency > duration:
            raise ValueError("Release frequency cannot exceed duration")
            
        self.duration = duration
        self.release_frequency = release_frequency
        
        self.release_epochs = []
        self.amounts = []
        self._initialize_schedule()
    
    def _initialize_schedule(self) -> None:
        num_releases = self.duration // self.release_frequency
        amount_per_release = self.initial_amount / num_releases
        
        for i in range(num_releases):
            epoch = self.start_epoch + (i * self.release_frequency)
            self.release_epochs.append(epoch)
            self.amounts.append(amount_per_release)

    def tokens_to_release(self, epoch: int) -> float:
        if epoch < self.start_epoch:
            return 0.0
        if epoch in self.release_epochs:
            index = self.release_epochs.index(epoch)
            return self.amounts[index]
        return 0.0

    def remaining_tokens(self, epoch: int) -> float:
        if epoch < self.start_epoch:
            return self.initial_amount
        released = sum(amount for e, amount in zip(self.release_epochs, self.amounts) 
                      if e <= epoch)
        return self.initial_amount - released