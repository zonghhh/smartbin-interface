"""
Timer Utilities for Smart Bin Interface
Handles countdown timers and timing operations.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Callable, Optional

class Timer:
    """Handles timer operations for the smart bin interface"""
    
    def __init__(self):
        self.active_timers = {}
        self.timer_threads = {}
    
    def start_countdown(self, 
                       timer_id: str, 
                       duration: int, 
                       callback: Optional[Callable] = None,
                       update_callback: Optional[Callable] = None) -> bool:
        """
        Start a countdown timer
        
        Args:
            timer_id: Unique identifier for the timer
            duration: Duration in seconds
            callback: Function to call when timer completes
            update_callback: Function to call on each second update
            
        Returns:
            bool: True if timer started successfully
        """
        if timer_id in self.active_timers:
            self.stop_timer(timer_id)
        
        self.active_timers[timer_id] = {
            'duration': duration,
            'remaining': duration,
            'start_time': time.time(),
            'callback': callback,
            'update_callback': update_callback,
            'active': True
        }
        
        # Start timer thread
        thread = threading.Thread(
            target=self._countdown_worker,
            args=(timer_id,),
            daemon=True
        )
        thread.start()
        self.timer_threads[timer_id] = thread
        
        return True
    
    def _countdown_worker(self, timer_id: str):
        """Worker thread for countdown timer"""
        while (timer_id in self.active_timers and 
               self.active_timers[timer_id]['active'] and
               self.active_timers[timer_id]['remaining'] > 0):
            
            time.sleep(1)
            
            if timer_id not in self.active_timers:
                break
            
            timer = self.active_timers[timer_id]
            timer['remaining'] -= 1
            
            # Call update callback if provided
            if timer['update_callback']:
                try:
                    timer['update_callback'](timer['remaining'])
                except Exception as e:
                    print(f"Update callback error: {e}")
            
            # Check if timer completed
            if timer['remaining'] <= 0:
                timer['active'] = False
                if timer['callback']:
                    try:
                        timer['callback']()
                    except Exception as e:
                        print(f"Timer callback error: {e}")
                break
    
    def stop_timer(self, timer_id: str) -> bool:
        """
        Stop a running timer
        
        Args:
            timer_id: Timer identifier
            
        Returns:
            bool: True if timer was stopped
        """
        if timer_id in self.active_timers:
            self.active_timers[timer_id]['active'] = False
            del self.active_timers[timer_id]
            
            if timer_id in self.timer_threads:
                del self.timer_threads[timer_id]
            
            return True
        return False
    
    def get_remaining_time(self, timer_id: str) -> int:
        """
        Get remaining time for a timer
        
        Args:
            timer_id: Timer identifier
            
        Returns:
            int: Remaining seconds, 0 if timer not found or completed
        """
        if timer_id in self.active_timers:
            return max(0, self.active_timers[timer_id]['remaining'])
        return 0
    
    def is_timer_active(self, timer_id: str) -> bool:
        """
        Check if a timer is active
        
        Args:
            timer_id: Timer identifier
            
        Returns:
            bool: True if timer is active
        """
        return (timer_id in self.active_timers and 
                self.active_timers[timer_id]['active'])
    
    def get_all_timers(self) -> dict:
        """
        Get all active timers
        
        Returns:
            dict: Dictionary of active timers
        """
        return self.active_timers.copy()
    
    def stop_all_timers(self):
        """Stop all active timers"""
        for timer_id in list(self.active_timers.keys()):
            self.stop_timer(timer_id)
    
    def pause_timer(self, timer_id: str) -> bool:
        """
        Pause a running timer
        
        Args:
            timer_id: Timer identifier
            
        Returns:
            bool: True if timer was paused
        """
        if timer_id in self.active_timers:
            self.active_timers[timer_id]['active'] = False
            return True
        return False
    
    def resume_timer(self, timer_id: str) -> bool:
        """
        Resume a paused timer
        
        Args:
            timer_id: Timer identifier
            
        Returns:
            bool: True if timer was resumed
        """
        if timer_id in self.active_timers:
            self.active_timers[timer_id]['active'] = True
            return True
        return False
    
    def reset_timer(self, timer_id: str, new_duration: int) -> bool:
        """
        Reset a timer with new duration
        
        Args:
            timer_id: Timer identifier
            new_duration: New duration in seconds
            
        Returns:
            bool: True if timer was reset
        """
        if timer_id in self.active_timers:
            self.stop_timer(timer_id)
            return self.start_countdown(timer_id, new_duration)
        return False

class CountdownDisplay:
    """Helper class for displaying countdown timers in Streamlit"""
    
    def __init__(self, timer: Timer):
        self.timer = timer
    
    def display_countdown(self, timer_id: str, placeholder=None):
        """
        Display countdown in Streamlit
        
        Args:
            timer_id: Timer identifier
            placeholder: Streamlit placeholder for display
        """
        remaining = self.timer.get_remaining_time(timer_id)
        
        if placeholder:
            if remaining > 0:
                placeholder.markdown(f"⏰ **{remaining}** seconds remaining")
            else:
                placeholder.markdown("⏰ **Time's up!**")
        else:
            if remaining > 0:
                st.markdown(f"⏰ **{remaining}** seconds remaining")
            else:
                st.markdown("⏰ **Time's up!**")
    
    def display_progress_bar(self, timer_id: str, total_duration: int, placeholder=None):
        """
        Display progress bar for countdown
        
        Args:
            timer_id: Timer identifier
            total_duration: Total duration in seconds
            placeholder: Streamlit placeholder for display
        """
        remaining = self.timer.get_remaining_time(timer_id)
        progress = (total_duration - remaining) / total_duration
        
        if placeholder:
            placeholder.progress(progress)
        else:
            st.progress(progress)

