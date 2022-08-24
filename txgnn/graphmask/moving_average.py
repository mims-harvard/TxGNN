class MovingAverage:

    window_size = None
    current_value = None
    observations = None
    observation_count = None
    use_window = None

    def __init__(self, window_size=None, use_window=True):
        self.window_size = window_size

        self.current_value = 0
        self.observation_count = 0
        self.use_window = use_window

        if use_window:
            self.observations = [None] * self.window_size

    def register(self, observation):
        if self.use_window:
            current_window_idx = self.observation_count % self.window_size
            insert_value = observation / self.window_size

        if self.use_window and self.observation_count >= self.window_size:
            drop_value = self.observations[current_window_idx]
            self.current_value -= drop_value
            self.current_value += insert_value
        else:
            div_val = self.observation_count + 1
            self.current_value *= self.observation_count
            self.current_value += observation
            self.current_value /= div_val

        if self.use_window:
            self.observations[current_window_idx] = insert_value

        self.observation_count += 1

    def get_value(self):
        return self.current_value
