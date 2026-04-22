import os
import csv
import time
from datetime import datetime

class DataLogger:
    def __init__(self, log_dir="logs", throttle_seconds=1.0):
        self.log_dir = log_dir
        self.throttle_seconds = throttle_seconds
        self.last_log_time = 0
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.violations_file = os.path.join(self.log_dir, "violations_log.csv")
        self.actions_file = os.path.join(self.log_dir, "action_log.csv")
        
        self._init_csv(self.violations_file, ["Timestamp", "Person1_ID", "Person2_ID"])
        self._init_csv(self.actions_file, ["Timestamp", "Person_ID", "Action"])

    def _init_csv(self, file_path, headers):
        # Always overwrite on initialization to start fresh every run
        with open(file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def should_log(self):
        """Check if enough time has passed to log again (throttling)."""
        current_time = time.time()
        if current_time - self.last_log_time >= self.throttle_seconds:
            return True
        return False

    def log_data(self, violations, actions):
        """Log data if throttling permits."""
        if not self.should_log():
            return
            
        current_time = time.time()
        self.last_log_time = current_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log Violations
        if violations:
            with open(self.violations_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                for v in violations:
                    # v is a tuple (idx1, idx2)
                    writer.writerow([timestamp, v[0], v[1]])
                    
        # Log Actions
        if actions:
            with open(self.actions_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                for person_idx, action in enumerate(actions):
                    if action and action != "unknown":
                        writer.writerow([timestamp, person_idx, action])
