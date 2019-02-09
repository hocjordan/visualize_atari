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

### score_frame

Produces an 'actor' saliency and a 'critic' saliency. Calls 'run_through_model' in either actor mode or critic mode. I think to get a baseline. Then loops through 80 (the size of the image) loops and for each loops it runs 'run_through_model' again. It then produces some kind of score by comparing the baseline 'run_through_model' to the specific loop's 'run_through_model'. This score provides a bitmap (effectively) that gives a score for every pixel in the atari game. The actor and critic scores outputted from this function are layered onto the original frame to produce the images in the paper.

For each mode, it operates as follows. First it calculates L using no mask. Then seems to run through all pixels (i, j) and for each pixel it generates a mask (a frame the same size as the visual input with blurring centred on the pixel of interest). Then calculates l using the mask. Then compares these to produces a score. I think that what this is doing is comparing the output of the actor/critic with no mask---i.e. usual visual input---against the output of the actor/critic with blurring centred on the pixel of interest. The relative output is then used to determine the importance of the pixel to the output.

### run_through_model

You take the history['ins'] which seems to be the observation space i.e. the visual input. If there is no mask (see score_frame), the visual input is unchanged. If there is a mask, the visual input is (I think) blurred using the mask. (Note: how the visual input is altered is determined by the interp_func, which is set to 'occlude'). The input, after the mask has been applied, is turned into a tensor and then into a 'Variable'. This is then fed into the model in some way (in the return line) to give the output of the actor or the critic based on that state.

### jacobian

Used for calculating Jacobian saliencies. Uses forward and backward 'hooks' and the output policy distribution. A 'hook' seems to be an indicator of sorts. A backward hook will be called every time the gradients with respect to module inputs are computed, and a forward hook is called every time an output has been computed.

## Variables

### hx, cx

Part of history, part of model. Might be LSTM activation and memory variables respectively. 

### top_dh_actor, top_dh_critic

Used for calculating jacobian saliencies.