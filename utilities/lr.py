# burn_in_scaling
def burn_in_scaling(cur_batch, burn_in, power):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Returns the darknet burn in scaling
    - Designed to be used with torch.optim.LambdaLR
    - Scaling formula is (cur_batch / burn_in)^power
    ----------
    """

    return pow(cur_batch / burn_in, power);

# LearnRateConstant
class LearnRateConstant:
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Handles scaling for the darknet "constant" policy
    - Designed to be used with torch.optim.LambdaLR (pass the step method)
    ----------
    """

    def __init__(self, batches_seen, burn_in, power):
        self.batches_seen = batches_seen
        self.burn_in = burn_in
        self.power = power

    def step(self, n):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Returns a scaling for the learn rate based on the darknet "constant" policy
        - Designed to be used with torch.optim.LambdaLR (pass this method as the lambda function)
        - If current batch number is less than burn_in, returns burn_in scale according to burn_in_scaling
        - Otherwise, returns 1.0 (in other words, just a constant learn rate)
        ----------
        """

        cur_batch = self.batches_seen + n + 1

        if(cur_batch < self.burn_in):
            scale = burn_in_scaling(cur_batch, self.burn_in, self.power)
        else:
            scale = 1.0

        return scale

# LearnRateSteps
class LearnRateSteps:
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Handles scaling for the darknet "steps" policy
    - Designed to be used with torch.optim.LambdaLR (pass the step method)
    ----------
    """

    def __init__(self, batches_seen, burn_in, power, steps, scales):
        self.batches_seen = batches_seen
        self.burn_in = burn_in
        self.power = power
        self.steps = steps
        self.scales = scales

    def step(self, n):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Returns a scaling for the learn rate based on the darknet "steps" policy
        - Designed to be used with torch.optim.LambdaLR (pass this method as the lambda function)
        - If current batch number is less than burn_in, returns burn_in scale according to burn_in_scaling
        - Otherwise, returns the steps scale where each scale is multiplied together if the
          batch number is >= the corresponding step
        ----------
        """

        cur_batch = self.batches_seen + n + 1

        if(cur_batch < self.burn_in):
            scale = burn_in_scaling(cur_batch, self.burn_in, self.power)
        else:
            scale = 1.0
            for i, step in enumerate(self.steps):
                if(cur_batch >= step):
                    scale *= self.scales[i]

        return scale
