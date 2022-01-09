import torch
from Model import NetModel
from Environment import Game2048
import numpy as np


def get_action(state, epsilon=0.):
    with torch.no_grad():
        q_values = model(state)[0].numpy()

    should_explore = np.random.binomial(1, epsilon)
    if should_explore:
        q_values = np.ones(q_values.shape[-1])

    q_values.astype(np.float64)
    q_values /= q_values.sum()

    return q_values


def compute_td_loss(states, actions, rewards, next_states, is_done, gamma=0.99, check_shapes=False):
    """ Compute td loss using torch operations only. Use the formula above. """
    states = states.view(-1, 16)
    next_states = next_states.view(-1, 16)
    actions = torch.tensor(actions, dtype=torch.long)  # shape: [batch_size]
    rewards = torch.tensor(rewards, dtype=torch.float32)  # shape: [batch_size]
    # shape: [batch_size, state_size]
    is_done = torch.tensor(is_done, dtype=torch.uint8)  # shape: [batch_size]

    # get q-values for all actions in current states
    predicted_qvalues = model(states)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[
        range(states.shape[0]), actions
    ]

    # compute q-values for all actions in next states
    with torch.no_grad():
        predicted_next_qvalues = model(next_states)

    # compute V*(next_states) using predicted next q-values
    next_state_values = torch.max(predicted_next_qvalues, dim=-1).values
    assert next_state_values.dtype == torch.float32

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    target_qvalues_for_actions = rewards + gamma * next_state_values

    # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    target_qvalues_for_actions = torch.where(
        is_done, rewards, target_qvalues_for_actions)

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions -
                       target_qvalues_for_actions) ** 2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim(
        ) == 2, "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim(
        ) == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim(
        ) == 1, "there's something wrong with target q-values, they must be a vector"

    return loss


def generate_session(t_max=2000, epsilon=0., train=False):
    """play env with approximate q-learning agent and train it at the same time"""
    total_reward = 0
    env = Game2048()
    s = env.table.to(torch.float32)

    for t in range(t_max):
        prob = get_action(s, epsilon=epsilon)
        is_move = False
        while not is_move:
            act = prob.argmax()
            is_move = env.action(act)
            if not is_move:
                prob += 1
                prob[act] = 0
                prob /= prob.sum()
            else:
                a = act

        next_s = env.table.to(torch.float32)
        r = env.get_reward()
        done = not env.is_played()

        if train:
            optimizer.zero_grad()
            compute_td_loss(s, [a], [r], next_s, [done], check_shapes=True).backward()
            optimizer.step()

        total_reward += r
        s = next_s
        if done:
            break

    return total_reward, env.get_score()


model = NetModel()
optimizer = torch.optim.Adam(model.parameters(), 5e-4)

# parameters
gamma = 0.99  # discount for MDP
epsilon = 0.5

for game in range(1000):
    session_rewards, scores = [], []
    data = [generate_session(epsilon=epsilon, train=True) for _ in range(100)]
    for reward, score in data:
        session_rewards.append(reward)
        scores.append(score)
    print("epoch #{}\tmean reward = {:.3f}\tscore = {:.3f}\tepsilon = {:.3f}".format(game, np.mean(session_rewards),
                                                                                     np.mean(scores), epsilon))

    epsilon *= 0.99
    assert epsilon >= 1e-4, "Make sure epsilon is always nonzero during training"
