import random


class BasicMemory:

    def __init__(self, memory_size=1000000):

        # how many training samples should we remember?
        self.memory_size = memory_size

        # the actual memory items
        self.memory = []

    def remember(self, state, p1_action, p2_action, p1_reward, p2_reward, state_prime, done):

        # remember everything we've been given
        self.memory.append((state, p1_action, p2_action, p1_reward, p2_reward, state_prime, done))

        # we don't want to remember it forever
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def sample_memory(self, count):

        # randomly sample from the memory
        samples = random.sample(self.memory, count)

        # pull apart the sample into its component parts
        states = [x[0] for x in samples]
        p1_actions = [x[1] for x in samples]
        p2_actions = [x[2] for x in samples]
        p1_rewards = [x[3] for x in samples]
        p2_rewards = [x[4] for x in samples]
        state_primes = [x[5] for x in samples]
        dones = [x[6] for x in samples]

        return states, p1_actions, p2_actions, p1_rewards, p2_rewards, state_primes, dones

    def num_samples(self):
        # how many samples do we have?
        return len(self.memory)



