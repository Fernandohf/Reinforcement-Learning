# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import Settings

PARAMETERS = {
    'BUFFER_SIZE': int(1e5),      # Replay buffer size
    'BATCH_SIZE': 128,            # Minibatch size
    'GAMMA': 1.0,                 # Discount factor
    'TAU': 1e-3,                  # Soft update of target parameters
    'LR_ACTOR': 1e-3,             # Learning rate of the actor
    'LR_CRITIC': 1e-3,            # Learning rate of the critic
    'WEIGHT_DECAY': .0001,        # L2 weight decay
    'UPDATE_EVERY_N_STEPS': 5,    # Number of step wait before update
    'UPDATE_N_TIMES': 10,         # Number of updates
    'GRADIENT_CLIP_VALUE': 2,     # Max gradient modulus for clipping
    # 'LR_STEP_SIZE': 30,         # LR step size
    # 'LR_GAMMA': .2,             # LR gamma multiplier
    'OU_THETA': .15,              # OU noise theta parameter
    'OU_SIGMA': .1,               # OU noise sigma parameters
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu")

}
DEFAULT_SETTINGS = Settings(PARAMETERS)


class MADDPG(DDPGAgent):
    """
    Implements a Multi Agent Deep Deterministic Policy Gradient.

    It uses a centralized training approach to a

    Parametes
    ----------
    settings: Settings object
        An object with all the hyperparameters for the model
        Requires the following value
    """
    def __init__(self, settings=DEFAULT_SETTINGS):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 14+2+2+2=20
        self.maddpg_agent = [DDPGAgent(),
                             DDPGAgent(14, 16, 8, 2, 20, 32, 16)]

        self.iter = 0

    def get_local_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor_local for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.actor_target for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    # def target_act(self, obs_all_agents, noise=0.0):
    #     """get target network actions from all the agents in the MADDPG object """
    #     target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
    #     return target_actions

    def update(self, samples, agent_number, logger):
        """
        Update the critics and actors of all the agents

        """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)

        obs_full = torch.stack(obs_full)
        next_obs_full = torch.stack(next_obs_full)

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)

        target_critic_input = torch.cat((next_obs_full.t(),target_actions), dim=1).to(device)

        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)

        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        action = torch.cat(action, dim=1)
        critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]

        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full.t(), q_input), dim=1)

        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)







