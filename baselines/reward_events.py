from datetime import datetime

class RewardEvents:
    def __init__(self) -> None:
        self.HasLeftOakLab = False
        
        # point values
        self.oaklab_points = 40

        # full point times in seconds
        self.oaklab_time = 30

        self.zero_point_time_multiplier = 3.0
    
    def apply_time_multiplier(self, points, time_goal, start_time):
        seconds_since_start = (datetime.now() - start_time).total_seconds() 

        if seconds_since_start > time_goal:
            return 0
        
        return points * (1 - (seconds_since_start / time_goal)) * self.zero_point_time_multiplier

    def calculate_reward(self, start_time):
        points = 0
        
        if self.HasLeftOakLab:
            points += self.apply_time_multiplier(self.oaklab_points, self.oaklab_time, start_time)

        return points
        