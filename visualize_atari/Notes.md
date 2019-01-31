# Greydanus code for visualising ATARI rl.

## major functions

### make_movie

Used to create movies including visual saliency. Has various arguments for gymp environments and movie settings (not really for RL though).

Sets up directory and gym environment. Sets up agent with (I think) a loaded policy.

Gets a rollout of the policy. I think this means to run whatever policy you current have for a given period of time (20 frames by default). Stores those frames as 'history'.

Make movie from frames in history. Seems to process saliencies during this phase: saliencies seem to be generated during rollout and stored in history. Now converted into images and overlaid on base frame. 

### rollout

Runs an episode of the environment for a given number of frames. Produces 'history' containing both base frames and saliencies.


### saliency_on_atari_frame

Seems to overlay saliencies onto a game frame to put into the movie. New versions of this function likely to be useful when adapting to new environments.

Note that this does just seem to be a simple function for taking a pre-existing image, blurring it slightly, and adding it to another image. It doesn't create the saliency map, it just merges it to the base frame.

### score frame

Produces an 'actor' saliency and a 'critic' saliency. Calls 'run_through_model' in either actor mode or critic mode. I think to get a baseline. Then loops through 80 (the size of the image) loops and for each loops it runs 'run_through_model' again. It then produces some kind of score by comparing the baseline 'run_through_model' to the specific loop's 'run_through_model'.

### run_through_model

You take the history['ins'] which seems to be the observation space i.e. the visual input, and I think you process that and put it back into the model.

### jacobian

Used for calculating Jacobian saliencies. Uses forward and backward 'hooks' and the output policy distribution. A 'hook' seems to be an indicator of sorts.

## Variables

### hx, cx

Part of history, part of model. Might be LSTM activation and memory variables respectively. 'The hook will be called every time the gradients with respect to module inputs are computed.'

### top_dh_actor, top_dh_critic

Used for calculating jacobian saliencies.